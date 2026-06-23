import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    input_dir: str
    output_dir: str
    thumbnail_size: int


def get_config() -> Config:
    return Config(
        input_dir=os.environ.get("INPUT_DIR", "/inputs"),
        output_dir=os.environ.get("OUTPUT_DIR", "/outputs"),
        thumbnail_size=int(os.environ.get("THUMBNAIL_SIZE", "256")),
    )
