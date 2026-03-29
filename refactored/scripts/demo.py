"""Demo viewer for Snake environment.

Two modes:
  1. Live display — watch the agent (or random policy) play in a window via OpenCV
  2. Record video — save an episode as MP4

Usage:
    # Live display with random agent
    python scripts/demo.py --mode live

    # Record a random episode to video
    python scripts/demo.py --mode record --output episode.mp4

    # Record side-by-side real vs imagined (once extraction pipeline exists)
    python scripts/demo.py --mode record --side-by-side --output comparison.mp4
"""

import argparse
import pathlib
import sys
import numpy as np
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from snake_env.snake_env import SnakeEnv  # pylint: disable=wrong-import-position
from constants import DISPLAY_SIZE, DEFAULT_FPS  # pylint: disable=wrong-import-position
from shared import upscale  # pylint: disable=wrong-import-position


def add_hud(display: np.ndarray, score: int, steps: int) -> np.ndarray:
    """Add score/step overlay to the display frame."""
    text = f"Score: {score}  Steps: {steps}"
    cv2.putText(
        display, text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    return display


def run_live(seed: int = 42, max_episodes: int = 10):
    """Display Snake in a live OpenCV window. Press Q to quit, R to restart."""
    env = SnakeEnv(max_steps=500)
    obs, info = env.reset(seed=seed)
    episode = 0

    print("Live demo — press Q to quit, any other key advances one step")
    print("Arrow keys: Up=w, Down=s, Left=a, Right=d (or any key for random)")

    while episode < max_episodes:
        frame_rgb = upscale(obs)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gs = info["game_state"]
        add_hud(frame_bgr, gs.score, gs.steps)
        cv2.imshow("Snake - DreamerV3 Demo", frame_bgr)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

        # Map keys to actions, default to random
        key_map = {ord("w"): 0, ord("d"): 1, ord("s"): 2, ord("a"): 3}
        action = key_map.get(key, env.action_space.sample())

        obs, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            gs = info["game_state"]
            print(f"Episode {episode + 1} ended — Score: {gs.score}, Steps: {gs.steps}")
            episode += 1
            obs, info = env.reset()

    cv2.destroyAllWindows()
    env.close()


def record_episode(
    output_path: str, seed: int = 42, max_steps: int = 500, fps: int = DEFAULT_FPS
):
    """Record a single episode to MP4 video."""
    env = SnakeEnv(max_steps=max_steps)
    obs, info = env.reset(seed=seed)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (DISPLAY_SIZE, DISPLAY_SIZE))

    frames = 0
    done = False
    while not done:
        frame_rgb = upscale(obs)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gs = info["game_state"]
        add_hud(frame_bgr, gs.score, gs.steps)
        writer.write(frame_bgr)

        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames += 1

    # Hold last frame for 1 second
    for _ in range(fps):
        writer.write(frame_bgr)

    writer.release()
    env.close()
    gs = info["game_state"]
    print(f"Recorded {frames} frames to {output_path}")
    print(f"Final score: {gs.score}, Steps: {gs.steps}")


def record_side_by_side(
    real_frames: np.ndarray,
    imagined_frames: np.ndarray,
    output_path: str,
    fps: int = DEFAULT_FPS,
):
    """Record side-by-side comparison of real vs imagined frames.

    Args:
        real_frames: (T, 64, 64, 3) uint8
        imagined_frames: (T, 64, 64, 3) uint8
        output_path: path for output MP4
    """
    width = DISPLAY_SIZE * 2 + 20  # two frames + gap
    height = DISPLAY_SIZE + 40  # frame + label space

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    n_frames = min(len(real_frames), len(imagined_frames))
    for i in range(n_frames):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Left: real frame
        real_up = upscale(real_frames[i])
        real_bgr = cv2.cvtColor(real_up, cv2.COLOR_RGB2BGR)
        canvas[30 : 30 + DISPLAY_SIZE, 0:DISPLAY_SIZE] = real_bgr
        cv2.putText(canvas, "Real", (DISPLAY_SIZE // 2 - 30, 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Right: imagined frame
        imag_up = upscale(imagined_frames[i])
        imag_bgr = cv2.cvtColor(imag_up, cv2.COLOR_RGB2BGR)
        x_offset = DISPLAY_SIZE + 20
        canvas[30 : 30 + DISPLAY_SIZE, x_offset : x_offset + DISPLAY_SIZE] = imag_bgr
        cv2.putText(canvas, "Imagined", (x_offset + DISPLAY_SIZE // 2 - 55, 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Step counter
        cv2.putText(canvas, f"Step {i}", (width // 2 - 40, height - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        writer.write(canvas)

    # Hold last frame
    for _ in range(fps):
        writer.write(canvas)

    writer.release()
    print(f"Recorded {n_frames} side-by-side frames to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake demo viewer")
    parser.add_argument("--mode", choices=["live", "record"], default="live")
    parser.add_argument("--output", default="episode.mp4", help="Output video path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    if args.mode == "live":
        run_live(seed=args.seed)
    elif args.mode == "record":
        record_episode(args.output, seed=args.seed, fps=args.fps)
