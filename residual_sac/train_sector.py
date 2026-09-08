"""Train a residual SAC policy on top of the sector racing-line controller.

The policy emits a bounded residual on two interpretable channels - reference speed and
steering - rather than raw pedals, and its mean head is zero-initialised, so training
starts at the controller's own score instead of from scratch.

A quadratic pull toward zero residual, decaying linearly over ``--penalty-decay-steps``,
holds the policy near the controller while the critic is still untrained. Without it the
random critic drags the zero-initialised actor off the controller within a few thousand
updates and the run collapses; that was measured on the pure-pursuit residual, where the
score fell from 84 to 15 inside 4000 steps.

The run checkpoints its whole state - networks, optimisers, replay buffer and random
streams - so a job killed by a time limit or a node failure resumes on the next
submission rather than restarting.

Usage::

    pixi run python residual_sac/train_sector.py --seed 0 --total-timesteps 700000 \
        --penalty-decay-steps 300000 --device cuda --resume
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

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

warnings.filterwarnings('ignore')

from agent_interface import SectorResidualAgent, convert_action, convert_obs  # noqa: E402
from pure_pursuit.rollout import make_env  # noqa: E402
from raceline.sector import DEFAULT_PARAMS, build_profile  # noqa: E402
from residual_sac.critic import QNetwork  # noqa: E402
from residual_sac.features_sector import FEATURE_DIM, residual_features  # noqa: E402
from residual_sac.policy import ResidualActor  # noqa: E402
from residual_sac.replay import ReplayBuffer  # noqa: E402
from util import save_model  # noqa: E402


def evaluate(env, agent, episodes=1):
    """
    Deterministic rollout through the graded path.

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


def penalty_at(step, args):
    """
    :return: the leash weight at a training step, decaying linearly to zero
    """
    return args.residual_penalty * max(
        0.0, 1.0 - step / max(args.penalty_decay_steps, 1))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total-timesteps', type=int, default=700_000)
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
    p.add_argument('--channels', type=int, default=2, choices=(2, 3),
                   help='2 for speed and steering, 3 to add the lateral aim-point shift')
    p.add_argument('--residual-penalty', type=float, default=1.0)
    p.add_argument('--penalty-decay-steps', type=int, default=300_000)
    p.add_argument('--warmup-noise', type=float, default=0.3)
    p.add_argument('--autotune', action='store_true', default=True)
    p.add_argument('--no-autotune', dest='autotune', action='store_false')
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--checkpoint-every', type=int, default=25_000)
    p.add_argument('--resume', action='store_true',
                   help='continue from the run directory checkpoint if one exists')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default=os.path.join(REPO_ROOT, 'runs_sector'))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    run_dir = os.path.join(args.out_dir, f'seed{args.seed}')
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, 'best.obj')
    ckpt_path = os.path.join(run_dir, 'checkpoint.pt')

    residual_dim = args.channels
    params = np.asarray(DEFAULT_PARAMS)
    profile = build_profile(params)
    line_args = (profile['line'], profile['s'], profile['speed'], profile['kappa'])

    env, eval_env = make_env(), make_env()

    actor = ResidualActor(FEATURE_DIM, args.hidden, residual_dim).to(device)
    q1 = QNetwork(FEATURE_DIM, residual_dim, args.hidden).to(device)
    q2 = QNetwork(FEATURE_DIM, residual_dim, args.hidden).to(device)
    q1_target = QNetwork(FEATURE_DIM, residual_dim, args.hidden).to(device)
    q2_target = QNetwork(FEATURE_DIM, residual_dim, args.hidden).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                             lr=args.lr_critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)

    log_alpha = torch.tensor(np.log(args.alpha), device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr_critic)
    target_entropy = -float(residual_dim)
    alpha = args.alpha

    buffer = ReplayBuffer(args.buffer_size, FEATURE_DIM, residual_dim)

    def snapshot(trained=True):
        """An agent carrying the actor's current weights, on CPU."""
        return SectorResidualAgent(*line_args, params, actor if trained else None,
                                   args.hidden, args.residual_scale, args.channels)

    start_step, history = 0, []
    baseline = None

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt['actor'])
        q1.load_state_dict(ckpt['q1'])
        q2.load_state_dict(ckpt['q2'])
        q1_target.load_state_dict(ckpt['q1_target'])
        q2_target.load_state_dict(ckpt['q2_target'])
        q_opt.load_state_dict(ckpt['q_opt'])
        actor_opt.load_state_dict(ckpt['actor_opt'])
        alpha_opt.load_state_dict(ckpt['alpha_opt'])
        with torch.no_grad():
            log_alpha.copy_(ckpt['log_alpha'].to(device))
        alpha = float(log_alpha.exp().item())
        buffer.obs = ckpt['buffer_obs']
        buffer.next_obs = ckpt['buffer_next_obs']
        buffer.actions = ckpt['buffer_actions']
        buffer.rewards = ckpt['buffer_rewards']
        buffer.dones = ckpt['buffer_dones']
        buffer.size, buffer.pos = ckpt['buffer_size'], ckpt['buffer_pos']
        start_step = ckpt['step']
        baseline, best_return = ckpt['baseline'], ckpt['best_return']
        history = ckpt['history']
        np.random.set_state(ckpt['np_rng'])
        torch.set_rng_state(ckpt['torch_rng'])
        print(f"resumed from step {start_step}, best {best_return:.3f}", flush=True)

    if baseline is None:
        baseline = evaluate(eval_env, snapshot(trained=False))
        best_return = baseline
        print(f"zero-residual baseline: {baseline:.3f}", flush=True)
        save_model(snapshot(), best_path)

    def write_checkpoint(step):
        """Atomic, so a job killed mid-write leaves the previous checkpoint intact."""
        tmp = ckpt_path + '.tmp'
        torch.save({'step': step,
                    'actor': actor.state_dict(),
                    'q1': q1.state_dict(), 'q2': q2.state_dict(),
                    'q1_target': q1_target.state_dict(),
                    'q2_target': q2_target.state_dict(),
                    'q_opt': q_opt.state_dict(),
                    'actor_opt': actor_opt.state_dict(),
                    'alpha_opt': alpha_opt.state_dict(),
                    'log_alpha': log_alpha.detach().cpu(),
                    'buffer_obs': buffer.obs, 'buffer_next_obs': buffer.next_obs,
                    'buffer_actions': buffer.actions, 'buffer_rewards': buffer.rewards,
                    'buffer_dones': buffer.dones,
                    'buffer_size': buffer.size, 'buffer_pos': buffer.pos,
                    'baseline': baseline, 'best_return': best_return,
                    'history': history,
                    'np_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(),
                    'args': vars(args)}, tmp)
        os.replace(tmp, ckpt_path)

    obs, _ = env.reset()
    features = residual_features(obs, snapshot().tracker(), params)
    rollout_agent = snapshot()
    ep_return, ep_len = 0.0, 0
    start = time.time()

    for step in range(start_step, args.total_timesteps):
        if step < args.learning_starts:
            residual = np.clip(np.random.normal(0.0, args.warmup_noise, residual_dim),
                               -1.0, 1.0).astype(np.float32)
        else:
            with torch.no_grad():
                sampled, _, _ = actor.sample(
                    torch.from_numpy(features).unsqueeze(0).to(device))
            residual = sampled.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(
            convert_action(rollout_agent.act_from_residual(obs, residual)))
        next_features = residual_features(next_obs, rollout_agent.tracker(), params)

        # Bootstrap through the 600-step time limit; only leaving the track is terminal.
        buffer.add(features, residual, reward, next_features, terminated)

        ep_return += reward
        ep_len += 1
        obs, features = next_obs, next_features

        if terminated or truncated:
            history.append({'step': step + 1, 'return': ep_return, 'length': ep_len})
            obs, _ = env.reset()
            features = residual_features(obs, rollout_agent.tracker(), params)
            ep_return, ep_len = 0.0, 0

        if step >= args.learning_starts:
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
                penalty = penalty_at(step, args)
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
            done_steps = step + 1 - start_step
            sps = done_steps / max(time.time() - start, 1e-9)
            print(f"step {step + 1:>8d}  eval {score:7.3f}  best {best_return:7.3f}  "
                  f"alpha {alpha:.4f}  penalty {penalty_at(step, args):.3f}  "
                  f"{sps:.0f} SPS", flush=True)
            if score > best_return:
                best_return = score
                save_model(candidate, best_path)
            rollout_agent = candidate

        if (step + 1) % args.checkpoint_every == 0:
            write_checkpoint(step + 1)

    write_checkpoint(args.total_timesteps)
    elapsed = time.time() - start
    with open(os.path.join(run_dir, 'summary.json'), 'w') as fh:
        json.dump({'seed': args.seed,
                   'baseline': baseline,
                   'best_return': best_return,
                   'total_timesteps': args.total_timesteps,
                   'penalty_decay_steps': args.penalty_decay_steps,
                   'channels': args.channels,
                   'wall_seconds': elapsed,
                   'args': vars(args),
                   'episodes': history[-200:]}, fh, indent=2)

    print(f"\nseed {args.seed}: best {best_return:.3f} from baseline {baseline:.3f} "
          f"in {elapsed / 3600:.2f} h -> {best_path}")


if __name__ == '__main__':
    main()
