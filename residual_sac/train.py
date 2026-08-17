"""Train a residual SAC policy on top of the pure-pursuit controller.

The policy emits a bounded residual on three interpretable channels rather than raw
pedals, and its mean head is zero-initialised, so training starts at the geometric
controller's score instead of from scratch.

A quadratic pull toward zero residual, decaying linearly over ``--penalty-decay-steps``,
holds the policy near the controller while the critic is still untrained. Without it the
random critic drags the zero-initialised actor off the controller within a few thousand
updates and the run collapses.

Usage::

    pixi run python residual_sac/train.py --seed 0 --total-timesteps 1000000 --device cuda
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

warnings.filterwarnings('ignore')

from agent_interface import ResidualAgent, convert_action, convert_obs  # noqa: E402
from pure_pursuit.controller import DEFAULT_PARAMS  # noqa: E402
from pure_pursuit.rollout import make_env  # noqa: E402
from residual_sac.critic import QNetwork  # noqa: E402
from residual_sac.features import FEATURE_DIM, RESIDUAL_DIM, residual_features  # noqa: E402
from residual_sac.policy import ResidualActor  # noqa: E402
from residual_sac.replay import ReplayBuffer  # noqa: E402
from util import save_model  # noqa: E402


def evaluate(env, agent, episodes=1):
    """
    Deterministic rollout, matching the graded policy.

    :return: mean episodic return
    """
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(
                convert_action(agent.get_action(convert_obs(obs))))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total-timesteps', type=int, default=1_000_000)
    p.add_argument('--buffer-size', type=int, default=300_000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-starts', type=int, default=5_000)
    p.add_argument('--lr-actor', type=float, default=3e-4)
    p.add_argument('--lr-critic', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--policy-frequency', type=int, default=2)
    p.add_argument('--target-frequency', type=int, default=1)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--residual-scale', type=float, default=1.0)
    p.add_argument('--residual-penalty', type=float, default=1.0)
    p.add_argument('--penalty-decay-steps', type=int, default=300_000)
    p.add_argument('--warmup-noise', type=float, default=0.3)
    p.add_argument('--autotune', action='store_true', default=True)
    p.add_argument('--no-autotune', dest='autotune', action='store_false')
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default=os.path.join(REPO_ROOT, 'runs'))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    run_dir = os.path.join(args.out_dir, f'seed{args.seed}')
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, 'best.obj')

    params = np.asarray(DEFAULT_PARAMS)
    env, eval_env = make_env(), make_env()

    actor = ResidualActor(FEATURE_DIM, args.hidden).to(device)
    q1 = QNetwork(FEATURE_DIM, RESIDUAL_DIM, args.hidden).to(device)
    q2 = QNetwork(FEATURE_DIM, RESIDUAL_DIM, args.hidden).to(device)
    q1_target = QNetwork(FEATURE_DIM, RESIDUAL_DIM, args.hidden).to(device)
    q2_target = QNetwork(FEATURE_DIM, RESIDUAL_DIM, args.hidden).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                             lr=args.lr_critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)

    log_alpha = torch.tensor(np.log(args.alpha), device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr_critic)
    target_entropy = -float(RESIDUAL_DIM)
    alpha = args.alpha

    buffer = ReplayBuffer(args.buffer_size, FEATURE_DIM, RESIDUAL_DIM)

    def snapshot():
        """An agent carrying the actor's current weights, on CPU."""
        return ResidualAgent(params, actor, args.hidden, args.residual_scale)

    baseline = evaluate(eval_env, ResidualAgent(params, None, args.hidden,
                                                args.residual_scale))
    print(f"zero-residual baseline: {baseline:.3f}", flush=True)
    best_return = baseline
    save_model(snapshot(), best_path)

    history = []
    obs, _ = env.reset()
    features = residual_features(obs, params)
    rollout_agent = snapshot()
    ep_return, ep_len = 0.0, 0
    start = time.time()

    for step in range(args.total_timesteps):
        if step < args.learning_starts:
            residual = np.clip(np.random.normal(0.0, args.warmup_noise, RESIDUAL_DIM),
                               -1.0, 1.0).astype(np.float32)
        else:
            with torch.no_grad():
                sampled, _, _ = actor.sample(
                    torch.from_numpy(features).unsqueeze(0).to(device))
            residual = sampled.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(
            convert_action(rollout_agent.act_from_residual(obs, residual)))
        next_features = residual_features(next_obs, params)

        # Bootstrap through the 600-step time limit; only leaving the track is terminal.
        buffer.add(features, residual, reward, next_features, terminated)

        ep_return += reward
        ep_len += 1
        obs, features = next_obs, next_features

        if terminated or truncated:
            history.append({'step': step + 1, 'return': ep_return, 'length': ep_len})
            obs, _ = env.reset()
            features = residual_features(obs, params)
            ep_return, ep_len = 0.0, 0

        if step < args.learning_starts:
            continue

        b_obs, b_act, b_rew, b_next, b_done = buffer.sample(args.batch_size, device)

        with torch.no_grad():
            next_act, next_logp, _ = actor.sample(b_next)
            target_q = torch.min(q1_target(b_next, next_act),
                                 q2_target(b_next, next_act)) - alpha * next_logp
            backup = b_rew + (1.0 - b_done) * args.gamma * target_q

        q_loss = (F.mse_loss(q1(b_obs, b_act), backup)
                  + F.mse_loss(q2(b_obs, b_act), backup))
        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_opt.step()

        if step % args.policy_frequency == 0:
            penalty = args.residual_penalty * max(
                0.0, 1.0 - step / max(args.penalty_decay_steps, 1))
            for _ in range(args.policy_frequency):
                pi, logp, _ = actor.sample(b_obs)
                actor_loss = (alpha * logp
                              - torch.min(q1(b_obs, pi), q2(b_obs, pi))).mean()
                if penalty > 0.0:
                    actor_loss = actor_loss + penalty * pi.pow(2).sum(-1).mean()
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()

                if args.autotune:
                    with torch.no_grad():
                        _, logp_detached, _ = actor.sample(b_obs)
                    alpha_loss = (-log_alpha.exp()
                                  * (logp_detached + target_entropy)).mean()
                    alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_opt.step()
                    alpha = log_alpha.exp().item()

        if step % args.target_frequency == 0:
            for net, target in ((q1, q1_target), (q2, q2_target)):
                for p, pt in zip(net.parameters(), target.parameters()):
                    pt.data.mul_(1.0 - args.tau).add_(args.tau * p.data)

        if (step + 1) % args.eval_every == 0:
            candidate = snapshot()
            score = evaluate(eval_env, candidate)
            sps = (step + 1) / (time.time() - start)
            penalty = args.residual_penalty * max(
                0.0, 1.0 - step / max(args.penalty_decay_steps, 1))
            print(f"step {step + 1:>8d}  eval {score:7.3f}  best {best_return:7.3f}  "
                  f"alpha {alpha:.4f}  penalty {penalty:.3f}  {sps:.0f} SPS", flush=True)
            if score > best_return:
                best_return = score
                save_model(candidate, best_path)
            rollout_agent = candidate

    elapsed = time.time() - start
    with open(os.path.join(run_dir, 'summary.json'), 'w') as fh:
        json.dump({'seed': args.seed,
                   'baseline': baseline,
                   'best_return': best_return,
                   'total_timesteps': args.total_timesteps,
                   'wall_seconds': elapsed,
                   'args': vars(args),
                   'episodes': history[-200:]}, fh, indent=2)

    print(f"\nseed {args.seed}: best {best_return:.3f} from baseline {baseline:.3f} "
          f"in {elapsed / 3600:.2f} h -> {best_path}")


if __name__ == '__main__':
    main()
