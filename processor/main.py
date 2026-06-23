import os
import logging

from processor.config import get_config
from processor.converter import generate_thumbnail

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

VERSION = "1.0.0"


def run():
    config = get_config()

    log.info(f"zarr-to-thumbnail processor version {VERSION}")
    log.info(f"Config: {config}")

    input_dirs = [
        f.path for f in os.scandir(config.input_dir)
        if f.is_dir() and f.name.endswith('.zarr')
    ]

    if not input_dirs:
        raise FileNotFoundError(f"No .zarr directories found in {config.input_dir}")

    for input_path in input_dirs:
        name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(config.output_dir, f"{name}_thumbnail.png")
        log.info(f"Processing: {input_path}")
        generate_thumbnail(input_path, output_path, config)
        log.info(f"Written: {output_path}")


if __name__ == "__main__":
    run()
