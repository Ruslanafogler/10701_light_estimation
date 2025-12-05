#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from UNLE.lightnet import LightNet
from UNLE.geometry_util import GeometryUtils
from unle import (
    physical_energy_and_intensity_torch,
    LightDatasetIAN
)
from util import load_dataset


# ============================================================
# Train LightNet with supervised + physics-guided loss
# ============================================================

def train_lightnet(
    model_name,
    epochs1,
    epochs2,
    batch_size,
    q,
    lr,
    results_dir
):

    print("\n=======================================")
    print(f" TRAINING LightNet on model={model_name}")
    print("=======================================\n")

    # ---------------------------------------------------------
    # Create directories
    # ---------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    # results_dir is project_dir / "results" / model_name / "unle"
    # So we need to go: results_dir -> model_name -> results -> project_dir -> dataset
    project_dir = results_dir.parent.parent.parent
    dataset_dir = project_dir / "dataset" / model_name

    print(f"Loading dataset from: {dataset_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    dataset = load_dataset(str(dataset_dir))

    # Convert RGB → grayscale
    rgb = dataset.rgb()                                 # (K,H_full,W_full,3)
    w = np.array([0.2126, 0.7152, 0.0722])
    images_gray_full = np.tensordot(rgb, w, axes=([3],[0]))  # (K,H,W)

    # Downsample to 128×128
    RES = 128
    K, Hf, Wf = images_gray_full.shape
    images_gray = np.zeros((K, RES, RES), np.float32)

    for k in range(K):
        images_gray[k] = cv2.resize(
            images_gray_full[k].astype(np.float32),
            (RES, RES),
            interpolation=cv2.INTER_AREA,
        )

    # ---------------------------------------------------------
    # Load UNLPS preprocessing results
    # ---------------------------------------------------------
    normals_unc = np.load(results_dir / "normals_unc.npy")
    albedos_unc = np.load(results_dir / "albedos_unc.npy")
    depth_unc   = np.load(results_dir / "depth_unc.npy")

    # Resize if needed
    if normals_unc.shape[:2] != (RES, RES):
        tmp = np.zeros((RES, RES, 3))
        for c in range(3):
            tmp[..., c] = cv2.resize(normals_unc[..., c], (RES, RES))
        normals_unc = tmp

    if albedos_unc.shape != (RES, RES):
        albedos_unc = cv2.resize(albedos_unc, (RES, RES))

    if depth_unc.shape != (RES, RES):
        depth_unc = cv2.resize(depth_unc, (RES, RES))

    mask = (albedos_unc > 0).astype(np.uint8)

    # ---------------------------------------------------------
    # Geometry for physics term
    # ---------------------------------------------------------
    cam = dataset.metadata["camera"]
    intr = {
        "fx": cam["fx"],
        "fy": cam["fy"],
        "cx": cam["cx"],
        "cy": cam["cy"],
    }

    X_unc = GeometryUtils.make_X_from_depth(depth_unc, intr)
    B_unc = albedos_unc[..., None] * normals_unc

    # Convert GT lights to camera frame
    light_w = dataset.light_positions()                # (K,3)
    R_cw = np.array(cam["R_cw"])
    if R_cw.ndim == 3:
        R_cw = R_cw[0]
    C = np.array(cam["location_world"])

    L_gt_cam = (R_cw @ (light_w - C).T).T              # (K,3)

    # ---------------------------------------------------------
    # Build dataset + dataloaders
    # ---------------------------------------------------------
    ds = LightDatasetIAN(images_gray, albedos_unc, normals_unc, L_gt_cam)
    loader_sup  = DataLoader(ds, batch_size=batch_size, shuffle=True)
    loader_phys = DataLoader(ds, batch_size=1, shuffle=True)

    # ---------------------------------------------------------
    # Model + optimizer
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = LightNet(in_channels=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_mse = nn.MSELoss()

    # Convert geometry to torch
    X_t = torch.from_numpy(X_unc).float().to(device)
    B_t = torch.from_numpy(B_unc).float().to(device)
    mask_t = torch.from_numpy(mask).bool().to(device)

    # ---------------------------------------------------------
    # CSV logger
    # ---------------------------------------------------------
    csv_path = log_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "phase",
            "supervised_loss",
            "physics_loss",
            "total_loss",
            "lr",
            "batch_size",
            "lambda_phys",
            "model"
        ])

    # =========================================================
    # PHASE 1 — SUPERVISED ONLY
    # =========================================================
    print("\n=== Phase 1: Supervised Only ===")
    for epoch in range(epochs1):
        model.train()
        running_loss = 0.0

        for x, y in loader_sup:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_mse(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(ds)
        print(f"[Phase1] Epoch {epoch+1}/{epochs1}  Loss={avg_loss:.6f}")

        # Log CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                "phase1",
                avg_loss,
                0,
                avg_loss,
                lr,
                batch_size,
                0,
                model_name
            ])

    # =========================================================
    # PHASE 2 — SUPERVISED + PHYSICS
    # =========================================================
    lambda_mse = 1.0
    lambda_phys = 0.5

    print("\n=== Phase 2: Supervised + Physics-Guided Loss ===")
    for epoch in range(epochs2):
        model.train()
        total_loss = 0.0
        total_phys = 0.0

        for x, y in loader_phys:
            x, y = x.to(device), y.to(device)

            I_k_t = x[0, 0]  # (H,W)

            optimizer.zero_grad()

            pred = model(x)  # (1,3)

            F_phys, e_k = physical_energy_and_intensity_torch(
                I_k_t, X_t, B_t, mask_t, pred, q
            )

            L_mse = loss_mse(pred, y)
            loss = lambda_mse * L_mse + lambda_phys * F_phys

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_phys += F_phys.item()

        avg_loss = total_loss / len(ds)
        avg_phys = total_phys / len(ds)
        print(f"[Phase2] Epoch {epoch+1}/{epochs2}  Total={avg_loss:.6f}  Phys={avg_phys:.6f}")

        # Log CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                "phase2",
                0,
                avg_phys,
                avg_loss,
                lr,
                batch_size,
                lambda_phys,
                model_name
            ])

    # ---------------------------------------------------------
    # Save final weights
    # ---------------------------------------------------------
    weights_path = results_dir / f"lightnet_{model_name}_physics.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"\nSaved model to {weights_path}")

    print("\nTraining complete.\n")


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sphere")
    parser.add_argument("--epochs_phase1", type=int, default=80)
    parser.add_argument("--epochs_phase2", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_dir = project_dir / "results" / args.model / "unle"

    train_lightnet(
        model_name=args.model,
        epochs1=args.epochs_phase1,
        epochs2=args.epochs_phase2,
        batch_size=args.batch_size,
        q=args.q,
        lr=args.lr,
        results_dir=results_dir
    )


if __name__ == "__main__":
    main()
