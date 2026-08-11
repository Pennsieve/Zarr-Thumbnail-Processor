import logging

import numpy as np
import zarr
from zarr.storage import LocalStore
from PIL import Image

from processor.config import Config

log = logging.getLogger(__name__)


def _build_slice_index(shape: tuple, axes: list[dict], is_rgb: bool) -> tuple:
    """Build a slice index to extract a single 2D XY slice from a zarr array.

    Returns a tuple that can be used to index the zarr array directly,
    so only the needed chunks are read from disk (not the whole volume).
    """
    axis_names = [a["name"].lower() for a in axes]
    log.info(f"Axes: {axis_names}, array shape: {shape}")

    idx = []
    for i, name in enumerate(axis_names):
        if name in ("x", "y"):
            idx.append(slice(None))
        elif name == "z":
            mid = shape[i] // 2
            log.info(f"Slicing Z axis (dim {i}) at index {mid}")
            idx.append(mid)
        elif name == "c" and is_rgb:
            idx.append(slice(None))  # keep all RGB channels
        else:
            # t, c (non-RGB), or any other leading dimension — take first
            idx.append(0)

    return tuple(idx)


def _postprocess_slice(slice_2d: np.ndarray, is_rgb: bool) -> np.ndarray:
    """Transpose RGB channels and drop alpha if needed."""
    if is_rgb and slice_2d.ndim == 3 and slice_2d.shape[0] in (3, 4):
        slice_2d = slice_2d.transpose(1, 2, 0)
        if slice_2d.shape[2] == 4:
            slice_2d = slice_2d[:, :, :3]

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
    omero = root.attrs.get("omero", {})
    omero_channels = omero.get("channels", [])
    # Detect RGB: 3 channels labeled R, G, B (color-agnostic)
    is_rgb = (
        len(omero_channels) == 3
        and {ch.get("label", "").upper() for ch in omero_channels} == {"R", "G", "B"}
    )
    log.info(f"Found {len(datasets)} resolution levels, is_rgb={is_rgb}")

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
    zarr_arr = root[chosen_level]

    log.info(f"Array shape: {zarr_arr.shape}, dtype: {zarr_arr.dtype}")

    # 3. Extract an XY slice — index the zarr array directly so only
    #    the needed chunks are read from disk, not the whole volume.
    if axes:
        idx = _build_slice_index(zarr_arr.shape, axes, is_rgb)
        slice_2d = zarr_arr[idx]
        if not isinstance(slice_2d, np.ndarray):
            slice_2d = np.array(slice_2d)
        slice_2d = _postprocess_slice(slice_2d, is_rgb)
    else:
        # Fallback: assume last two dims are Y, X (OME-Zarr convention)
        extra = zarr_arr.ndim - 2
        idx = tuple(s // 2 if i < extra else slice(None) for i, s in enumerate(zarr_arr.shape))
        slice_2d = zarr_arr[idx]
        if not isinstance(slice_2d, np.ndarray):
            slice_2d = np.array(slice_2d)
        log.info(f"No axes metadata; assumed last two dims are YX, shape: {slice_2d.shape}")

    # 4. Normalize to 0–255 uint8
    smin, smax = float(slice_2d.min()), float(slice_2d.max())
    if smax - smin > 0:
        normalized = (slice_2d - smin) / (smax - smin) * 255.0
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.float64)
    img_array = normalized.astype(np.uint8)

    # 5. Center-crop to square, then resize to thumbnail_size
    pil_mode = "RGB" if is_rgb and img_array.ndim == 3 else "L"
    img = Image.fromarray(img_array, mode=pil_mode)
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((config.thumbnail_size, config.thumbnail_size), Image.LANCZOS)

    # 6. Save as PNG
    img.save(output_path, format="PNG")
    log.info(f"Thumbnail saved: {output_path} ({img.size[0]}x{img.size[1]})")
