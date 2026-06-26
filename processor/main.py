import os
import logging

from processor.config import get_config
from processor.converter import generate_thumbnail

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

VERSION = "2.0.0"


def _find_zarr_stores(input_dir: str) -> list[tuple[str, str]]:
    """Find zarr stores in input_dir. Returns list of (path, name) tuples.

    Supports two layouts:
    - INPUT_DIR is itself a zarr store (has zarr.json at root)
    - INPUT_DIR contains zarr subdirectories (e.g. file1.zarr/, file2.zarr/)
    """
    # Check if input_dir itself is a zarr store
    if os.path.exists(os.path.join(input_dir, "zarr.json")):
        name = os.path.basename(input_dir.rstrip("/"))
        # Strip .zarr extension if present
        if name.endswith(".zarr"):
            name = name[:-5]
        return [(input_dir, name)]

    # Otherwise scan for subdirectories containing zarr.json
    stores = []
    for entry in os.scandir(input_dir):
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, "zarr.json")):
            name = entry.name
            if name.endswith(".zarr"):
                name = name[:-5]
            stores.append((entry.path, name))

    return stores


def run():
    config = get_config()

    log.info(f"zarr-to-thumbnail processor version {VERSION}")
    log.info(f"Config: {config}")

    stores = _find_zarr_stores(config.input_dir)
    if not stores:
        raise FileNotFoundError(f"No zarr stores found in {config.input_dir}")

    log.info(f"Found {len(stores)} zarr store(s)")

    for zarr_path, name in stores:
        output_path = os.path.join(config.output_dir, f"{name}_thumbnail.png")
        log.info(f"Processing: {zarr_path} -> {output_path}")
        generate_thumbnail(zarr_path, output_path, config)
        log.info(f"Written: {output_path}")


if __name__ == "__main__":
    run()
