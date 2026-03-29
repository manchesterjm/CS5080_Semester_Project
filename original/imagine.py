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

from ruamel.yaml import YAML as _YAML
import models
import tools


def load_config(config_name="snake"):
    """Load and merge default + named config from configs.yaml.

    Args:
        config_name: Config section name (e.g. 'snake', 'snake_32x32', 'snake_16x16')
    """
    yaml = _YAML(typ="safe", pure=True)
    config_path = DREAMER_DIR / "configs.yaml"
    configs = yaml.load(config_path.read_text())

    # Merge defaults + named config (same as dreamer.py)
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                recursive_update(base[key], value)
            else:
                base[key] = value

    merged = dict(configs["defaults"])
    recursive_update(merged, configs[config_name])

    # Convert to namespace (same as argparse does in dreamer.py)
    parsed = {}
    for key, value in merged.items():
        parsed[key] = tools.args_type(value)(value) if isinstance(value, str) else value
    config = argparse.Namespace(**parsed)

    # Derived values that main() normally computes
    config.num_actions = 4  # Snake has 4 discrete actions
    return config


def build_world_model(config):
    """Construct a WorldModel with the correct obs/act spaces."""
    import gym

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


@torch.no_grad()
def imagine_episode(wm, episode, context_length, device):
    """Run imagination on a single episode.

    1. Feed first `context_length` frames through encoder + dynamics.observe()
    2. Imagine forward for remaining timesteps using real actions
    3. Decode imagined latent states to RGB frames

    Returns dict with real_frames, imagined_frames, actions, rewards, etc.
    """
    T = len(episode["image"])
    imagine_length = T - context_length
    if imagine_length <= 0:
        return None

    # Build batch-of-1 data dict matching preprocess() expectations
    data = {
        "image": torch.tensor(
            episode["image"], dtype=torch.float32, device=device
        ).unsqueeze(0),  # (1, T, 64, 64, 3)
        "action": torch.tensor(
            episode["action"], dtype=torch.float32, device=device
        ).unsqueeze(0),  # (1, T, 4)
        "is_first": torch.tensor(
            episode["is_first"].astype(np.float32), device=device
        ).unsqueeze(0),  # (1, T)
        "is_terminal": torch.tensor(
            episode["is_terminal"].astype(np.float32), device=device
        ).unsqueeze(0),  # (1, T)
        "reward": torch.tensor(
            episode["reward"], dtype=torch.float32, device=device
        ).unsqueeze(0),  # (1, T)
    }

    # Preprocess (normalizes image to 0-1, adds cont)
    data = wm.preprocess(data)

    # Encode all frames
    embed = wm.encoder(data)  # (1, T, embed_dim)

    # Observe first context_length frames → posterior states
    context_embed = embed[:, :context_length]
    context_action = data["action"][:, :context_length]
    context_is_first = data["is_first"][:, :context_length]

    states, _ = wm.dynamics.observe(context_embed, context_action, context_is_first)

    # Take last posterior as init state for imagination
    init = {k: v[:, -1] for k, v in states.items()}

    # Imagine forward using real actions from remaining timesteps
    future_actions = data["action"][:, context_length:]
    prior = wm.dynamics.imagine_with_action(future_actions, init)

    # Decode imagined states to frames
    feat = wm.dynamics.get_feat(prior)
    imagined_decoded = wm.heads["decoder"](feat)["image"].mode()  # (1, T-ctx, 64, 64, 3)

    # Also decode context states for reconstruction quality check
    context_feat = wm.dynamics.get_feat(states)
    context_decoded = wm.heads["decoder"](context_feat)["image"].mode()  # (1, ctx, 64, 64, 3)

    # Convert to uint8 numpy
    imagined_frames = (imagined_decoded[0].clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
    context_recon = (context_decoded[0].clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
    real_frames = episode["image"][context_length:]  # already uint8

    return {
        "real_frames": real_frames,           # (T-ctx, 64, 64, 3) uint8
        "imagined_frames": imagined_frames,   # (T-ctx, 64, 64, 3) uint8
        "context_recon": context_recon,       # (ctx, 64, 64, 3) uint8
        "context_real": episode["image"][:context_length],  # (ctx, 64, 64, 3) uint8
        "actions": episode["action"][context_length:],      # (T-ctx, 4) float32
        "rewards": episode["reward"][context_length:],      # (T-ctx,) float32
        "is_terminal": episode["is_terminal"][context_length:],  # (T-ctx,) bool
        "episode_length": T,
        "context_length": context_length,
    }


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

    # Load config and build world model
    config = load_config(config_name)
    config.device = device
    config.compile = False  # Don't compile for inference

    wm = build_world_model(config).to(device)
    wm = load_checkpoint_into_wm(wm, checkpoint_path, device)

    # Load eval episodes
    min_length = context_length + 5  # Need at least 5 imagination steps
    episodes = load_eval_episodes(eval_dir, max_episodes=num_episodes * 2,
                                  min_length=min_length)

    # Process episodes
    all_results = []
    processed = 0
    for ep in episodes:
        if processed >= num_episodes:
            break

        # Cap episode length for memory
        total_len = min(len(ep["image"]), context_length + max_imagine_steps)
        ep_trimmed = {k: v[:total_len] for k, v in ep.items()}

        result = imagine_episode(wm, ep_trimmed, context_length, device)
        if result is not None:
            all_results.append(result)
            processed += 1

    if not all_results:
        print(f"  WARNING: No episodes processed for {checkpoint_path}")
        return None

    # Save as NPZ — store each episode's data with indexed keys
    save_dict = {"num_episodes": np.array(len(all_results))}
    for i, r in enumerate(all_results):
        for key, val in r.items():
            save_dict[f"ep{i}_{key}"] = np.array(val)

    np.savez_compressed(output_file, **save_dict)
    print(f"  Saved {len(all_results)} episodes to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Extract imagined trajectories from DreamerV3 checkpoints")
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
