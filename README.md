# Phase DOE: TIE4 + Physics-Guided CNN / Fourier / Tensor Train

Docker-ready solution for phase-only 128×128 diffractive optical elements.

## What the program returns
For each input set of 4 target images (`z1.png` ... `z4.png`), the program writes:
- `doe.png` — the main submission file, 128×128, 8-bit grayscale, phase-coded with `0 -> -pi`, `255 -> +pi`
- `doe_reconstruction_z1.png` ... `doe_reconstruction_z4.png` — reconstructed intensities at the same four distances used by the algorithm
- `doe_metrics.json` — per-plane and mean `SSIM` / `Pearson`
- `doe_history.csv` — optimization history

## Important
- The model is **not pretrained**.
- For every new 4-image case, the network is initialized **from scratch** with a fixed seed and optimized for up to **100 steps**.
- Input intensities are read **without normalization**.

## Build
```bash
docker build -t doe-submission:latest .
```

## Run one case
Input folder must contain `z1.png`, `z2.png`, `z3.png`, `z4.png`.

```bash
mkdir -p output
docker run --rm \
  -v "$(pwd)/examples/symbols:/input:ro" \
  -v "$(pwd)/output:/output" \
  doe-submission:latest \
  --input-dir /input --output /output/doe.png --steps 100 --nn-downsample 2 --device cpu
```

## Run many cases
```bash
mkdir -p output
docker run --rm \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  doe-submission:latest \
  --input-root /input --output /output --steps 100 --nn-downsample 2 --device cpu
```

Each nested case directory gets its own outputs and independent random initialization.

## Method
1. TIE4 initialization from the 4 target images.
2. Backward propagation to the DOE plane.
3. Physics-guided optimization with CNN + Fourier + TT.
4. Export the best 8-bit DOE PNG.

The repository also supports `--propagator fresnel` (default, for contest compatibility) and `--propagator asm` (research mode).

## Output format
Main file `doe.png`:
- PNG
- 128×128
- 8-bit grayscale (`L`)

Auxiliary files do **not** replace `doe.png`; they are returned for inspection because the algorithm also computes reconstructions and metrics internally.

## Repository contents
```text
Dockerfile
requirements.txt
README.md
run.py
src/doe/
examples/symbols/z1.png ... z4.png
tests/
notebooks/research_reference.ipynb
```
