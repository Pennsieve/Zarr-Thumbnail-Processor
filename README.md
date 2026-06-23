# zarr-to-thumbnail

Pennsieve workflow processor that generates a 256x256 PNG thumbnail from an OME-Zarr store.

## How it works

1. Scans the input directory for `.zarr` directories
2. Reads OME-Zarr multiscale metadata to find available resolution levels
3. Loads the **lowest resolution level** (smallest data, fastest)
4. Extracts the **middle slice** along axis 0
5. Normalizes pixel values to 0–255 and resizes to the configured thumbnail size
6. Saves as PNG to the output directory

## Configuration

| Environment Variable | Default    | Description                    |
|---------------------|------------|--------------------------------|
| `INPUT_DIR`         | `/inputs`  | Directory containing .zarr dirs |
| `OUTPUT_DIR`        | `/outputs` | Directory for output PNGs       |
| `THUMBNAIL_SIZE`    | `256`      | Width & height in pixels        |

## Usage

### Local

```bash
# Place .zarr directories in test_inputs/
make local
# Output PNGs appear in test_outputs/
```

### Docker

```bash
make run
```

### Lambda

The handler accepts camelCase event keys: `inputDir`, `outputDir`, `thumbnailSize`.

## Output

For each `<name>.zarr` in the input directory, produces `<name>_thumbnail.png` in the output directory.
