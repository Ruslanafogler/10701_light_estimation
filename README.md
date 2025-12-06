# 10701 Light Source Estimation

Team: Shengxi Wu, Steven Lee, Ruslana Fogler, Priyanka Vijaybaskar

## About 

Calibrated Photometric Stereo is a popular algorithm used to recover depth maps by relying on images and known directions of light sources in a scene that illuminate an object. In our 10701 project, we aim to solve the underdetermined *inverse* problem of Uncalibrated Photometric Stereo: given input images of an object being illuminated by many light sources, one at a time for each image, we would like to solve for (1) albedo + normal maps (pseudonormals) and (2) the positions of the light sources (near-field variant).

## Setup

> [!WARNING] 
> Only tested on Linux/MacOS.

Install `uv` to manage Python dependencies (if not already installed): 
```sh 
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then run:
```sh
uv sync
source .venv/bin/activate
```
This creates a venv with all required dependencies. (To add new dependencies, instead of `pip install <pkg-name>` you can run `uv add <pkg-name>` or the equivalent `uv pip install <pkg-name>`).

## Project Structure

```
10701_light_estimation/
├── src/
│   ├── UNLE/                    # Uncalibrated Near-Light Estimation
│   │   ├── light_estimation.py  # Light position optimization
│   │   ├── lightnet.py          # ML-based light estimator
│   │   ├── depth_integrator.py  # Depth integration from normals
│   │   └── geometry_util.py     # Geometry utilities
│   ├── calibrated_ff.py         # Calibrated far-field photometric stereo
│   ├── calibrated_nf.py         # Calibrated near-field photometric stereo
│   ├── uncalibrated_ff.py       # Uncalibrated far-field photometric stereo
│   ├── unle.py                  # UNLPS engine + LightNet training
│   ├── experiment_lightnet.py   # Full LightNet experiment pipeline
│   ├── dataset.py               # Dataset generation with Mitsuba
│   ├── gradient_domain.py       # Gradient-domain methods
│   ├── photometric.py           # Evaluation metrics
│   ├── visualize.py             # Visualization utilities
│   └── Poisson_Depth_Recovery/  # Poisson-based depth recovery
├── demo.ipynb                   # Main demo notebook
├── demo_unle_lightnet.ipynb     # UNLPS + LightNet demo
└── compare_lightnet_gt.py       # Compare LightNet predictions with GT

```

## Usage

Use [Jupyter Notebook](demo.ipynb) to generate dataset, run experiments, and perform evaluations.

### Train LightNet

Train LightNet for light position estimation:

```bash
cd src
python experiment_lightnet.py \
    --model sphere \
    --lr 1e-4 \
    --batch 8 \
    --epochs1 200 \
    --epochs2 80 \
    --lambda_phys 0.1 \
    --q 3 \
    --K 200
```

**Arguments:**
- `--model`: Dataset name (sphere, bunny, etc.)
- `--lr`: Learning rate
- `--batch`: Batch size
- `--epochs1`: Supervised-only training epochs
- `--epochs2`: Physics-guided training epochs
- `--lambda_phys`: Physics loss weight
- `--q`: Near-light falloff exponent (should always use 3)
- `--K`: Number of lights
- `--resume`: Path to checkpoint to resume training (optional)
