import os
import logging

from processor.config import get_config
from processor.converter import generate_thumbnail

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

VERSION = "1.1.0"


def run():
    config = get_config()

    log.info(f"zarr-to-thumbnail processor version {VERSION}")
    log.info(f"Config: {config}")

    # The input directory IS the zarr store (contains zarr.json directly),
    # not a parent directory containing .zarr subdirectories.
    zarr_json = os.path.join(config.input_dir, "zarr.json")
    if not os.path.exists(zarr_json):
        raise FileNotFoundError(f"No zarr.json found in {config.input_dir} — expected a zarr store")

    output_path = os.path.join(config.output_dir, "thumbnail.png")
    log.info(f"Processing: {config.input_dir}")
    generate_thumbnail(config.input_dir, output_path, config)
    log.info(f"Written: {output_path}")


if __name__ == "__main__":
    run()
