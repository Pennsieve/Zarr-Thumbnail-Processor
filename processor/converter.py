import logging

import numpy as np
import zarr
from zarr.storage import LocalStore
from PIL import Image

from processor.config import Config

log = logging.getLogger(__name__)


def _extract_xy_slice(arr: np.ndarray, axes: list[dict]) -> np.ndarray:
    """Extract a single 2D XY slice from a multi-dimensional OME-Zarr array.

    Uses the axes metadata to identify dimension roles (t, c, z, y, x) and
    picks the middle index for any non-XY spatial dimension, channel 0, and
    time 0.
    """
    axis_names = [a["name"].lower() for a in axes]
    log.info(f"Axes: {axis_names}, array shape: {arr.shape}")

    # Build an index tuple: middle for z, first for t/c, full slice for x/y
    idx = []
    for i, name in enumerate(axis_names):
        if name in ("x", "y"):
            idx.append(slice(None))
        elif name == "z":
            mid = arr.shape[i] // 2
            log.info(f"Slicing Z axis (dim {i}) at index {mid}")
            idx.append(mid)
        else:
            # t, c, or any other leading dimension — take first
            idx.append(0)

    slice_2d = arr[tuple(idx)]
    log.info(f"Extracted XY slice, shape: {slice_2d.shape}")
    return slice_2d


def generate_thumbnail(input_path: str, output_path: str, config: Config) -> None:
    """Generate a PNG thumbnail from an OME-Zarr store."""
    # 1. Open the zarr store and read multiscale metadata
    store = LocalStore(input_path)
    root = zarr.open_group(store, mode="r")

    multiscales = root.attrs.get("multiscales")
    if not multiscales:
        raise ValueError(f"No OME-Zarr multiscale metadata found in {input_path}")

    datasets = multiscales[0]["datasets"]
    axes = multiscales[0].get("axes", [])
    log.info(f"Found {len(datasets)} resolution levels")

    # 2. Pick the smallest resolution level that is >= thumbnail size
    #    Fall back to the largest (first) level if none are big enough.
    chosen_level = datasets[0]["path"]
    for ds in reversed(datasets):
        path = ds["path"]
        level_arr = root[path]
        # Last two dims are Y, X by OME-Zarr convention
        y_size, x_size = level_arr.shape[-2], level_arr.shape[-1]
        if min(y_size, x_size) >= config.thumbnail_size:
            chosen_level = path
            break
    log.info(f"Using resolution level: {chosen_level}")
    arr = root[chosen_level][:]

    log.info(f"Array shape: {arr.shape}, dtype: {arr.dtype}")

    # 3. Extract an XY slice using axes metadata
    if axes:
        slice_2d = _extract_xy_slice(arr, axes)
    else:
        # Fallback: assume last two dims are Y, X (OME-Zarr convention)
        extra = arr.ndim - 2
        idx = tuple(s // 2 if i < extra else slice(None) for i, s in enumerate(arr.shape))
        slice_2d = arr[idx]
        log.info(f"No axes metadata; assumed last two dims are YX, shape: {slice_2d.shape}")

    # 4. Normalize to 0–255 uint8
    smin, smax = float(slice_2d.min()), float(slice_2d.max())
    if smax - smin > 0:
        normalized = (slice_2d - smin) / (smax - smin) * 255.0
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.float64)
    img_array = normalized.astype(np.uint8)

    # 5. Center-crop to square, then resize to thumbnail_size
    img = Image.fromarray(img_array, mode="L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((config.thumbnail_size, config.thumbnail_size), Image.LANCZOS)

    # 6. Save as PNG
    img.save(output_path, format="PNG")
    log.info(f"Thumbnail saved: {output_path} ({img.size[0]}x{img.size[1]})")
