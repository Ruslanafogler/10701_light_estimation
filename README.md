# 10701 Light Source Estimation

Team: Shengxi Wu, Steven Lee, Ruslana Fogler, Priyanka Vijaybaskar

## About 

Calibrated Photometric Stereo is a popular algorithm used to recover depth maps by relying on images and known directions of light sources in a scene that illuminate an object. In our 10701 project, we aim to solve the underdetermined *inverse* problem of Uncalibrated Photometric Stereo: given input images of an object being illuminated by many light sources, one at a time for each image, we would like to solve for (1) albedo + normal maps (pseudonormals) and (2) the orientation of the light source (direction, and potentially position if we choose to pursue the near-field variant of the problem).

## Setup

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

## Dataset Generation
We will be using a synthetic dataset for this project. To generate the dataset, run 
```
python src/gen_dataset.py
``` 
This will create a folder `/dataset` in the project root containing dataset images.
