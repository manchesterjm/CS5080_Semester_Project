"""Imagination extraction from DreamerV3 checkpoints.

For a given checkpoint, loads the world model, feeds real eval episodes through
encode→observe→imagine→decode, and saves paired real/imagined frames.

Usage:
    python imagine.py --checkpoint path/to/checkpoint.pt --episodes 20 --context 5
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

# Add dreamerv3-torch to path for model imports
DREAMER_DIR = pathlib.Path(__file__).parent / "dreamerv3-torch"
sys.path.insert(0, str(DREAMER_DIR))

import models  # pylint: disable=import-error,wrong-import-position

from shared import load_config  # pylint: disable=wrong-import-position


def build_world_model(config):
    """Construct a WorldModel with the correct obs/act spaces."""
    import gym  # pylint: disable=import-outside-toplevel

    h, w = config.size
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
        "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
    })

    step = 0
    wm = models.WorldModel(obs_space, config.num_actions, step, config)
    return wm


def load_checkpoint_into_wm(wm, checkpoint_path, device):
    """Load checkpoint weights into world model, stripping _orig_mod prefix."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent_sd = checkpoint["agent_state_dict"]

    # Extract world model keys, strip _orig_mod. prefix from torch.compile
    wm_prefix = "_wm._orig_mod."
    wm_sd = {}
    for key, value in agent_sd.items():
        if key.startswith(wm_prefix):
            new_key = key[len(wm_prefix):]
            wm_sd[new_key] = value

    wm.load_state_dict(wm_sd)
    wm.eval()
    return wm


def load_eval_episodes(eval_dir, max_episodes=None, min_length=10):
    """Load eval episodes from NPZ files, filtering by minimum length."""
    eval_dir = pathlib.Path(eval_dir)
    episodes = []

    for npz_path in sorted(eval_dir.glob("*.npz")):
        ep = dict(np.load(npz_path))
        if len(ep["image"]) >= min_length:
            episodes.append(ep)
            if max_episodes and len(episodes) >= max_episodes:
                break

    return episodes


def _build_episode_tensors(episode, device):
    """Convert numpy episode arrays to batched torch tensors."""
    return {
        "image": torch.tensor(
            episode["image"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "action": torch.tensor(
            episode["action"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "is_first": torch.tensor(
            episode["is_first"].astype(np.float32), device=device
        ).unsqueeze(0),
        "is_terminal": torch.tensor(
            episode["is_terminal"].astype(np.float32), device=device
        ).unsqueeze(0),
        "reward": torch.tensor(
            episode["reward"], dtype=torch.float32, device=device
        ).unsqueeze(0),
    }


def _observe_and_imagine(wm, data, context_length):
    """Encode, observe context, and imagine forward from last posterior."""
    embed = wm.encoder(data)
    ctx = slice(None, context_length)
    states, _ = wm.dynamics.observe(
        embed[:, ctx], data["action"][:, ctx], data["is_first"][:, ctx],
    )
    init = {k: v[:, -1] for k, v in states.items()}
    future_actions = data["action"][:, context_length:]
    prior = wm.dynamics.imagine_with_action(future_actions, init)
    return states, prior


def _decode_to_uint8(wm, latent_states):
    """Decode latent states to uint8 RGB frames."""
    feat = wm.dynamics.get_feat(latent_states)
    decoded = wm.heads["decoder"](feat)["image"].mode()
    return (decoded[0].clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def imagine_episode(wm, episode, context_length, device):
    """Run imagination on a single episode.

    Returns dict with real_frames, imagined_frames, actions, rewards, etc.
    """
    total = len(episode["image"])
    if total - context_length <= 0:
        return None
    data = _build_episode_tensors(episode, device)
    data = wm.preprocess(data)
    states, prior = _observe_and_imagine(wm, data, context_length)
    return {
        "real_frames": episode["image"][context_length:],
        "imagined_frames": _decode_to_uint8(wm, prior),
        "context_recon": _decode_to_uint8(wm, states),
        "context_real": episode["image"][:context_length],
        "actions": episode["action"][context_length:],
        "rewards": episode["reward"][context_length:],
        "is_terminal": episode["is_terminal"][context_length:],
        "episode_length": total,
        "context_length": context_length,
    }


def _build_world_model_from_checkpoint(checkpoint_path, config_name, device):
    """Load config, build world model, and load checkpoint weights."""
    config = load_config(config_name)
    config.device = device
    config.compile = False
    wm = build_world_model(config).to(device)
    return load_checkpoint_into_wm(wm, checkpoint_path, device)


def _save_imagination_results(all_results, output_file):
    """Save imagination results as compressed NPZ."""
    save_dict = {"num_episodes": np.array(len(all_results))}
    for i, r in enumerate(all_results):
        for key, val in r.items():
            save_dict[f"ep{i}_{key}"] = np.array(val)
    np.savez_compressed(output_file, **save_dict)
    print(f"  Saved {len(all_results)} episodes to {output_file}")


def run_imagination(checkpoint_path, eval_dir, output_dir, num_episodes=20,
                    context_length=5, device="cuda:0", max_imagine_steps=60,
                    config_name="snake"):
    """Run imagination extraction for a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        eval_dir: Directory containing eval episode .npz files
        output_dir: Directory to save output
        num_episodes: Number of episodes to process
        context_length: Number of observed frames before imagination
        device: torch device
        max_imagine_steps: Cap imagination length to prevent memory issues
        config_name: Config section name from configs.yaml
    """
    output_dir = pathlib.Path(output_dir)
    output_file = output_dir / "imagination_data.npz"
    if output_file.exists():
        print(f"  Output already exists: {output_file}")
        return output_file

    output_dir.mkdir(parents=True, exist_ok=True)
    wm = _build_world_model_from_checkpoint(checkpoint_path, config_name, device)

    min_length = context_length + 5
    episodes = load_eval_episodes(eval_dir, max_episodes=num_episodes * 2,
                                  min_length=min_length)

    all_results = []
    for ep in episodes:
        if len(all_results) >= num_episodes:
            break
        total_len = min(len(ep["image"]), context_length + max_imagine_steps)
        ep_trimmed = {k: v[:total_len] for k, v in ep.items()}
        result = imagine_episode(wm, ep_trimmed, context_length, device)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print(f"  WARNING: No episodes processed for {checkpoint_path}")
        return None

    _save_imagination_results(all_results, output_file)
    return output_file


def main():
    """Parse arguments and run imagination extraction for one checkpoint."""
    parser = argparse.ArgumentParser(
        description="Extract imagined trajectories from DreamerV3 checkpoints"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--eval-dir", type=str,
                        default="training_output/snake_1M/eval_eps",
                        help="Directory containing eval episode .npz files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: analysis_output/checkpoint_NNNNNNN/)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to process")
    parser.add_argument("--context", type=int, default=5, help="Context length (observed frames)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--max-steps", type=int, default=60,
                        help="Max imagination steps per episode")
    args = parser.parse_args()

    # Derive output dir from checkpoint name if not specified
    if args.output_dir is None:
        ckpt_name = pathlib.Path(args.checkpoint).stem  # e.g. checkpoint_0555000
        args.output_dir = f"analysis_output/{ckpt_name}"

    print(f"Processing checkpoint: {args.checkpoint}")
    run_imagination(
        checkpoint_path=args.checkpoint,
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        context_length=args.context,
        device=args.device,
        max_imagine_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
