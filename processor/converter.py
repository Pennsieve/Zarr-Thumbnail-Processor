import logging

import numpy as np
import zarr
from zarr.storage import LocalStore
from PIL import Image

from processor.config import Config

log = logging.getLogger(__name__)


def generate_thumbnail(input_path: str, output_path: str, config: Config) -> None:
    """Generate a PNG thumbnail from an OME-Zarr store."""
    # 1. Open the zarr store and read multiscale metadata
    store = LocalStore(input_path)
    root = zarr.open_group(store, mode="r")

    multiscales = root.attrs.get("multiscales")
    if not multiscales:
        raise ValueError(f"No OME-Zarr multiscale metadata found in {input_path}")

    datasets = multiscales[0]["datasets"]
    log.info(f"Found {len(datasets)} resolution levels")

    # 2. Use the lowest resolution level (last in the list)
    lowest_level = datasets[-1]["path"]
    log.info(f"Using lowest resolution level: {lowest_level}")
    arr = root[lowest_level][:]

    log.info(f"Array shape: {arr.shape}, dtype: {arr.dtype}")

    # 3. Extract the middle slice along axis 0
    mid = arr.shape[0] // 2
    slice_2d = arr[mid, :, :]
    log.info(f"Extracted middle slice at index {mid}, shape: {slice_2d.shape}")

    # 4. Normalize to 0–255 uint8
    smin, smax = float(slice_2d.min()), float(slice_2d.max())
    if smax - smin > 0:
        normalized = (slice_2d - smin) / (smax - smin) * 255.0
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.float64)
    img_array = normalized.astype(np.uint8)

    # 5. Resize to thumbnail_size x thumbnail_size
    img = Image.fromarray(img_array, mode="L")
    img = img.resize((config.thumbnail_size, config.thumbnail_size), Image.LANCZOS)

    # 6. Save as PNG
    img.save(output_path, format="PNG")
    log.info(f"Thumbnail saved: {output_path} ({config.thumbnail_size}x{config.thumbnail_size})")
