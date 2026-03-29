"""Watch a trained DreamerV3 agent play Snake.

Modes:
  live   — real-time OpenCV window (default). Press Q to quit, SPACE to pause.
  record — save episodes to MP4 video file.

Usage:
    cd /home/josh/cs5080_project
    source .venv/bin/activate
    python scripts/play_trained.py                              # live, 10 fps
    python scripts/play_trained.py --fps 15                     # faster playback
    python scripts/play_trained.py --mode record --episodes 5   # save to video
"""

import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "dreamerv3-torch"))

# pylint: disable=wrong-import-position,wrong-import-order
from snake_env.snake_env import SnakeEnv
from constants import (
    DISPLAY_SCALE, DISPLAY_SIZE, DEFAULT_FPS, NUM_ACTIONS,
    IMAGE_SIZE, CELL_SIZE, PADDING, MAX_PLAY_STEPS,
)
from shared import upscale, load_config

import tools  # pylint: disable=import-error
from dreamer import Dreamer  # pylint: disable=import-error
CHECKPOINT_PATH = str(PROJECT_DIR / "training_output" / "snake_1M" / "latest.pt")


def load_agent(config, checkpoint_path):
    """Create Dreamer agent and load trained weights."""
    import gym  # pylint: disable=import-outside-toplevel

    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
    })
    act_space = gym.spaces.Discrete(NUM_ACTIONS)
    act_space.discrete = True

    pathlib.Path("/tmp/dreamer_eval").mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(pathlib.Path("/tmp/dreamer_eval"), 0)

    agent = Dreamer(obs_space, act_space, config, logger, iter([]))
    agent = agent.to(config.device)
    agent.requires_grad_(False)

    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    # Strip _orig_mod. prefixes from compiled model checkpoint keys
    raw_state = checkpoint["agent_state_dict"]
    cleaned_state = {k.replace("._orig_mod.", "."): v for k, v in raw_state.items()}
    agent.load_state_dict(cleaned_state)
    agent.eval()

    return agent


def add_hud(display, score, steps, episode):
    """Overlay score/step/episode info."""
    text = f"Ep {episode}  Score: {score}  Steps: {steps}"
    cv2.putText(
        display, text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    return display


def make_obs(image, is_first=False):
    """Construct batched observation dict for the agent."""
    return {
        "image": image[np.newaxis],
        "is_first": np.array([is_first], dtype=np.float32),
        "is_last": np.array([False], dtype=np.float32),
        "is_terminal": np.array([False], dtype=np.float32),
    }


def get_action(agent, obs, is_first, state):
    """Run agent policy, return discrete action and new state."""
    obs_dict = make_obs(obs, is_first=is_first)
    with torch.no_grad():
        policy_output, state = agent._policy(obs_dict, state, training=False)  # pylint: disable=protected-access
    action = int(torch.argmax(policy_output["action"][0]).item())
    return action, state


def window_closed(name):
    """Check if the user closed the window via the X button."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _determine_death(env, terminated):
    """Return (death_reason, ghost_cell) from the last game step."""
    if not terminated:
        return "TIMEOUT", None
    game = env.game
    dr, dc = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[game.direction]
    ghost = (game.head[0] + dr, game.head[1] + dc)
    r, c = ghost
    if r < 0 or r >= game.grid_size or c < 0 or c >= game.grid_size:
        return "WALL", ghost
    return "SELF", ghost


def _highlight_cell(frame_bgr, row, col):
    """Draw a red rectangle on the given grid cell in the display frame."""
    y0 = (PADDING + row * CELL_SIZE) * DISPLAY_SCALE
    x0 = (PADDING + col * CELL_SIZE) * DISPLAY_SCALE
    cell_px = CELL_SIZE * DISPLAY_SCALE
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + cell_px, y0 + cell_px),
                  (0, 0, 255), -1)


def _render_death_frame(obs, gs, ep, env, terminated, death_reason, ghost):
    """Render the death/timeout overlay frame and return it."""
    frame_bgr = cv2.cvtColor(upscale(obs), cv2.COLOR_RGB2BGR)
    add_hud(frame_bgr, gs.score, gs.steps, ep + 1)
    if terminated and death_reason == "SELF":
        _highlight_cell(frame_bgr, ghost[0], ghost[1])
    elif terminated and death_reason == "WALL":
        _highlight_cell(frame_bgr, env.game.head[0], env.game.head[1])
    cv2.putText(frame_bgr, death_reason,
                (DISPLAY_SIZE // 2 - 60, DISPLAY_SIZE // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(frame_bgr, f"Score: {gs.score}",
                (DISPLAY_SIZE // 2 - 80, DISPLAY_SIZE // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame_bgr


def _handle_pause(window_name):
    """Block until space/q is pressed or window closed. Return True to quit."""
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord(" ") or key == ord("q") or window_closed(window_name):
            break
    return key == ord("q") or window_closed(window_name)


def _run_single_episode(agent, env, seed, ep, delay_ms, window_name):
    """Play one episode, returning (score, quit_requested)."""
    obs, info = env.reset(seed=seed + ep)
    state, is_first, done = None, True, False
    while not done:
        frame_bgr = cv2.cvtColor(upscale(obs), cv2.COLOR_RGB2BGR)
        gs = info["game_state"]
        add_hud(frame_bgr, gs.score, gs.steps, ep + 1)
        cv2.imshow(window_name, frame_bgr)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q") or window_closed(window_name):
            return gs.score, True
        if key == ord(" ") and _handle_pause(window_name):
            return gs.score, True
        action, state = get_action(agent, obs, is_first, state)
        is_first = False
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    gs = info["game_state"]
    death_reason, ghost = _determine_death(env, terminated)
    print(f"  Episode {ep + 1}: Score {gs.score}, Steps {gs.steps} ({death_reason})")
    death_frame = _render_death_frame(obs, gs, ep, env, terminated, death_reason, ghost)
    cv2.imshow(window_name, death_frame)
    for _ in range(30):
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q") or window_closed(window_name):
            return gs.score, True
    return gs.score, False


def run_live(agent, episodes, seed, fps):
    """Display Snake in a real-time OpenCV window."""
    env = SnakeEnv(max_steps=MAX_PLAY_STEPS)
    delay_ms = max(1, int(1000 / fps))
    window_name = "DreamerV3 Snake Agent"
    scores = []
    print(f"Live display at {fps} FPS — Q to quit, SPACE to pause/resume, or close window")
    for ep in range(episodes):
        score, quit_req = _run_single_episode(agent, env, seed, ep, delay_ms, window_name)
        scores.append(score)
        if quit_req:
            break
    cv2.destroyAllWindows()
    env.close()
    return scores


def run_record(agent, episodes, output, seed, fps):
    """Record episodes to MP4 video."""
    env = SnakeEnv(max_steps=MAX_PLAY_STEPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (DISPLAY_SIZE, DISPLAY_SIZE))
    scores = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        state = None
        is_first = True
        done = False

        while not done:
            frame_bgr = cv2.cvtColor(upscale(obs), cv2.COLOR_RGB2BGR)
            gs = info["game_state"]
            add_hud(frame_bgr, gs.score, gs.steps, ep + 1)
            writer.write(frame_bgr)

            action, state = get_action(agent, obs, is_first, state)
            is_first = False

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        gs = info["game_state"]
        scores.append(gs.score)
        print(f"  Episode {ep + 1}: Score {gs.score}, Steps {gs.steps}")

        # Hold game over frame for 1.5 seconds
        frame_bgr = cv2.cvtColor(upscale(obs), cv2.COLOR_RGB2BGR)
        add_hud(frame_bgr, gs.score, gs.steps, ep + 1)
        cv2.putText(
            frame_bgr, "GAME OVER", (DISPLAY_SIZE // 2 - 100, DISPLAY_SIZE // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
        )
        for _ in range(int(fps * 1.5)):
            writer.write(frame_bgr)

    writer.release()
    env.close()
    print(f"\nRecorded {episodes} episodes to {output}")
    return scores


def main():
    """Parse arguments and launch live or recorded Snake agent playback."""
    parser = argparse.ArgumentParser(description="Watch trained DreamerV3 Snake agent")
    parser.add_argument("--mode", choices=["live", "record"], default="live")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", default="trained_agent.mp4")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    print("Loading config...")
    config = load_config()
    config.compile = False
    config.logdir = "/tmp/dreamer_eval"

    if "cuda" in config.device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"

    print(f"Loading agent from {args.checkpoint}...")
    agent = load_agent(config, args.checkpoint)
    print(f"Agent loaded on {config.device}\n")

    if args.mode == "live":
        scores = run_live(agent, args.episodes, args.seed, args.fps)
    else:
        scores = run_record(agent, args.episodes, args.output, args.seed, args.fps)

    print(f"\nScores: {scores}")
    print(f"Average: {np.mean(scores):.1f}, Best: {max(scores)}, Worst: {min(scores)}")


if __name__ == "__main__":
    main()
