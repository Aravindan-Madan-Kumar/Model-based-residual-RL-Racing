"""Record the shipped agent driving one evaluation episode as an animated GIF.

    pixi run python raceline/scripts/record_race.py
    pixi run python raceline/scripts/record_race.py --width 1280 --out assets/agent_race.gif

The drive loop mirrors ``try_agent.py`` exactly (``convert_obs`` -> ``get_action``
-> ``convert_action``), so the clip shows the same behaviour the grader scores.
Frames come from the headless cairo renderer (``mode="rgb_array"``); no display and
no ffmpeg are needed. Pillow writes the GIF directly.
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.chdir(REPO_ROOT)  # BirdView loads steering_wheel.png relative to cwd

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from agent_interface import convert_action, convert_obs  # noqa: E402
from util import create_env, load_model  # noqa: E402

EXPECTED_RETURN = 97.9  # try_agent.py mean for the shipped SectorResidualAgent


def rollout(model, env, seed, width):
    """Drive one episode, returning (frames, episodic_return).

    ``frames`` is a list of HxWx3 uint8 RGB arrays, one per step.
    """
    height = round(9 * width / 16)
    obs, _ = env.reset(seed=seed)
    frames, total, done = [], 0.0, False
    while not done:
        with torch.no_grad():
            action = convert_action(model.get_action(convert_obs(obs)))
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
        frames.append(env.unwrapped.render(mode='rgb_array', width=width, height=height))
    return frames, total


def encode_gif(frames, out, fps, colors, scale=1.0):
    """Write ``frames`` to ``out`` as a looping GIF, return the file size in bytes."""
    imgs = []
    for f in frames:
        im = Image.fromarray(f)
        if scale != 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)))
        imgs.append(im.quantize(colors=colors, method=Image.MEDIANCUT))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=round(1000 / fps), loop=0, optimize=True)
    return os.path.getsize(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='./models/model.obj')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--fps', type=float, default=10.0)
    p.add_argument('--out', default='assets/agent_race.gif')
    p.add_argument('--colors', type=int, default=256)
    p.add_argument('--max-mb', type=float, default=30.0,
                   help='shrink the clip until it fits under this size')
    p.add_argument('--no-check', action='store_true',
                   help='skip the episodic-return sanity assertion')
    p.add_argument('--frames-cache', default=None,
                   help='npz path: load rendered frames from here if it exists, '
                        'otherwise roll out once and write it (skips the ~10 min '
                        'rollout on re-encodes)')
    args = p.parse_args()

    if args.frames_cache and os.path.exists(args.frames_cache):
        data = np.load(args.frames_cache)
        frames, ret = list(data['frames']), float(data['ret'])
        print(f'loaded {len(frames)} cached frames (return {ret:.3f})')
    else:
        model = load_model(args.model)
        env = create_env(seed=args.seed, render_env=False, render_width=args.width)
        try:
            frames, ret = rollout(model, env, args.seed, args.width)
        finally:
            env.close()
        if args.frames_cache:
            np.savez_compressed(args.frames_cache, frames=np.asarray(frames), ret=ret)

    print(f'episode: {len(frames)} steps, return {ret:.3f} '
          f'(expected ~{EXPECTED_RETURN})')
    if not args.no_check:
        assert abs(ret - EXPECTED_RETURN) < 1.5, (
            f'return {ret:.3f} is far from the expected {EXPECTED_RETURN}; '
            f'the drive path looks broken - not writing a clip')

    # First pass at requested quality, then degrade in steps until it fits.
    # Resolution is given up before frame rate - smooth motion showcases better
    # than a sharp but choppy clip.
    passes = [
        dict(frames=frames, fps=args.fps, colors=args.colors, scale=1.0, note='full'),
        dict(frames=frames, fps=args.fps, colors=128, scale=0.75,
             note='0.75x, 128 colours'),
        dict(frames=frames, fps=args.fps, colors=128, scale=0.6,
             note='0.6x, 128 colours'),
        dict(frames=frames, fps=args.fps, colors=128, scale=0.5,
             note='0.5x, 128 colours'),
        dict(frames=frames[::2], fps=args.fps / 2, colors=128, scale=0.6,
             note='0.6x, every 2nd frame, 128 colours'),
    ]
    for i, cfg in enumerate(passes):
        mb = encode_gif(cfg['frames'], args.out, cfg['fps'], cfg['colors'],
                        cfg['scale']) / 1e6
        print(f'pass {i} ({cfg["note"]}): {mb:.1f} MB')
        if mb <= args.max_mb:
            break

    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, '
          f'{len(frames)} source frames)')


if __name__ == '__main__':
    main()
