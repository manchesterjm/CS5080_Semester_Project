"""Extract game state from RGB Snake frames at any resolution.

Uses the known color scheme to identify head, body, and food positions
by sampling the center pixel of each grid cell and matching to the
nearest known color.

Works on both crisp real frames (exact match) and blurry imagined frames
(nearest-neighbor color matching with thresholds).

Supports 64x64 (default), 32x32, 16x16, and other resolutions by
computing cell size and padding from the frame dimensions.
"""

from typing import NamedTuple, Optional
import numpy as np

# Reference colors (RGB, 0-255)
COLORS = {
    "background": np.array([0, 0, 0], dtype=np.float32),
    "head": np.array([0, 255, 0], dtype=np.float32),
    "body": np.array([0, 128, 0], dtype=np.float32),
    "food": np.array([255, 0, 0], dtype=np.float32),
}

# Default rendering constants for 64x64 (match snake_env.py)
CELL_SIZE = 6
PADDING = 2


class ExtractedState(NamedTuple):
    """State extracted from an RGB frame."""
    head: Optional[tuple[int, int]]          # (row, col) or None if not found
    body: list[tuple[int, int]]              # body segment positions (no head)
    food: Optional[tuple[int, int]]          # (row, col) or None if not found
    cell_labels: np.ndarray                  # (grid_size, grid_size) label grid


def _compute_cell_geometry(image_size: int, grid_size: int):
    """Compute cell size and padding for a given image and grid size.

    The 64x64 rendering uses CELL_SIZE=6, PADDING=2 for a 10x10 grid
    (6*10 + 2*2 = 64). For resized images, the geometry scales proportionally
    via nearest-neighbor, so we compute the effective cell/padding sizes.
    """
    if image_size == 64 and grid_size == 10:
        return CELL_SIZE, PADDING
    # For resized images: pixel positions scale by (new_size / 64)
    scale = image_size / 64.0
    cell_size = CELL_SIZE * scale
    padding = PADDING * scale
    return cell_size, padding


class StateExtractor:
    """Parse RGB Snake frames into game state.

    For real frames, extraction is exact (colors are perfectly distinct).
    For imagined/decoded frames, uses nearest-color matching with a
    configurable distance threshold.
    """

    def __init__(self, grid_size: int = 10, max_color_distance: float = 100.0):
        self.grid_size = grid_size
        self.max_color_distance = max_color_distance
        # Pre-compute color matrix for vectorized matching
        self._color_names = ["background", "head", "body", "food"]
        self._color_matrix = np.stack(
            [COLORS[name] for name in self._color_names]
        )  # (4, 3)

    def extract(self, frame: np.ndarray) -> ExtractedState:
        """Extract game state from an RGB frame at any resolution.

        Args:
            frame: (H, W, 3) uint8 RGB image (64x64, 32x32, 16x16, etc).

        Returns:
            ExtractedState with head, body, food positions and label grid.
        """
        h, w = frame.shape[:2]
        cell_size, padding = _compute_cell_geometry(h, self.grid_size)

        frame_f = frame.astype(np.float32)
        labels = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)

        head = None
        body = []
        food = None

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Sample center pixel of cell
                cy = int(padding + r * cell_size + cell_size / 2)
                cx = int(padding + c * cell_size + cell_size / 2)
                cy = min(cy, h - 1)
                cx = min(cx, w - 1)
                pixel = frame_f[cy, cx]

                # Find nearest reference color
                distances = np.linalg.norm(
                    self._color_matrix - pixel[np.newaxis, :], axis=1
                )
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]

                if best_dist > self.max_color_distance:
                    labels[r, c] = 0  # default to background
                    continue

                labels[r, c] = best_idx
                label_name = self._color_names[best_idx]

                if label_name == "head":
                    head = (r, c)
                elif label_name == "body":
                    body.append((r, c))
                elif label_name == "food":
                    food = (r, c)

        return ExtractedState(
            head=head,
            body=body,
            food=food,
            cell_labels=labels,
        )

    def compare(
        self, real_head: tuple[int, int], real_body: list[tuple[int, int]],
        real_food: tuple[int, int], extracted: ExtractedState
    ) -> dict:
        """Compare ground truth game state against extracted state.

        Returns dict with:
            head_error: Euclidean distance between head positions (grid cells)
            body_accuracy: fraction of body segments correctly placed (0-1)
            food_correct: bool
            head_found: bool
            food_found: bool
        """
        # Head position error
        if extracted.head is not None:
            head_error = float(np.linalg.norm(
                np.array(real_head) - np.array(extracted.head)
            ))
            head_found = True
        else:
            head_error = float(self.grid_size)  # max possible error
            head_found = False

        # Body segment accuracy
        real_body_set = set(real_body)
        extracted_body_set = set(extracted.body)
        if len(real_body_set) > 0:
            body_accuracy = len(real_body_set & extracted_body_set) / len(real_body_set)
        else:
            body_accuracy = 1.0

        # Food accuracy
        food_correct = (extracted.food is not None) and (extracted.food == real_food)
        food_found = extracted.food is not None

        return {
            "head_error": head_error,
            "body_accuracy": body_accuracy,
            "food_correct": food_correct,
            "head_found": head_found,
            "food_found": food_found,
        }
