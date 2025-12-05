#!/usr/bin/env python3
"""
Unified LightNet Runner:
- Loads dataset + UNLPS preprocessing
- Trains (supervised → supervised+physics)
- Validates each epoch
- Tests automatically at end
- Logs all metrics
- Saves loss and validation curves
- Saves final weights

Usage:
    python run_lightnet_full.py --model sphere
"""

import argparse
from pathlib import Path
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from UNLE.lightnet import LightNet
from UNLE.geometry_util import GeometryUtils
from unle import LightDatasetIAN, physical_energy_and_intensity_torch
from util import load_dataset


def physics_error(I_k, X, B, mask, L, q):
    mask_flat = mask.ravel()
    Xf = X.reshape(-1,3)[mask_flat]
    Bf = B.reshape(-1,3)[mask_flat]
    If = I_k.ravel()[mask_flat]

    v = L[None,:] - Xf
    r = np.maximum(np.linalg.norm(v,axis=1), 1e-8)
    phi = np.sum(Bf * v, axis=1) / (r**q)

    denom = np.sum(phi*phi) + 1e-8
    ek = np.sum(If * phi) / denom
    return float(np.mean((If - ek*phi)**2))


def train_one_epoch(model, loader_sup, loader_phys, optimizer, loss_fn,
                    X_t, B_t, mask_t, lambda_phys, q, device):

    sup_loss_sum = 0
    phys_loss_sum = 0
    total_loss_sum = 0
    n = 0


    model.train()
    for x, y in loader_sup:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss_sup = loss_fn(pred, y)
        loss_sup.backward()
        optimizer.step()

        sup_loss_sum += loss_sup.item() * x.size(0)
        total_loss_sum += loss_sup.item() * x.size(0)
        n += x.size(0)

    # Physics pass``
    for x, y in loader_phys:
        x, y = x.to(device), y.to(device)
        I_k = x[0,0]  # (H,W)

        optimizer.zero_grad()
        pred = model(x)
        F_phys, _ = physical_energy_and_intensity_torch(I_k, X_t, B_t, mask_t, pred, q)
        loss = loss_fn(pred, y) + lambda_phys * F_phys
        loss.backward()
        optimizer.step()

        phys_loss_sum += F_phys.item()
        total_loss_sum += loss.item()
        n += 1

    return (
        sup_loss_sum / n,
        phys_loss_sum / n,
        total_loss_sum / n
    )


def validate(model, X_val, L_val, device):
    model.eval()
    with torch.no_grad():
        pred = model(X_val).cpu().numpy()

    eu = []
    for i in range(len(pred)):
        eu.append(np.linalg.norm(pred[i] - L_val[i]))

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Ground Truth
    ax.scatter(L_val[:, 0], L_val[:, 1], L_val[:, 2], color='g', label="Ground Truth Light Positions", s=35, alpha=0.8)
    # Prediction
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], color='b', label="Estimated Light Positions", marker='^', s=35, alpha=0.8)

    for i in range(len(pred)):
        ax.plot([L_val[i, 0], pred[i, 0]], [L_val[i, 1], pred[i, 1]], [L_val[i, 2], pred[i, 2]], 'k--', alpha=0.4, linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Light Positions: Estimated vs Ground Truth")
    ax.legend()
    plt.tight_layout()
    plt.savefig("light_positions_comparison_3d.png")
    plt.show()
    plt.close()

    return float(np.mean(eu))


def run_experiment(model_name, lr, batch, epochs1, epochs2, lambda_phys, q=3, K=100, resume=False):
    print("\n==============================")
    print(" Running LightNet Experiment")
    print("==============================\n")

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Output directory
    out_dir = project_dir / "experiments" / model_name / f"lr{lr}_b{batch}_lam{lambda_phys}_K{K}_512"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DATASET
    train_dir = project_dir / "dataset" / model_name
    test_dir = project_dir / "dataset" / "test" / model_name
    dataset = load_dataset(str(train_dir))
    test_dataset = load_dataset(str(test_dir))

    rgb = dataset.rgb()
    w = np.array([0.2126, 0.7152, 0.0722])
    gray_full = np.tensordot(rgb, w, axes=([3],[0]))

    test_rgb = test_dataset.rgb()
    test_w = np.array([0.2126, 0.7152, 0.0722])
    test_gray_full = np.tensordot(test_rgb, test_w, axes=([3],[0])) 


    RES = 128
    test_gray = np.stack([cv2.resize(test_gray_full[k], (RES,RES)) for k in range(test_rgb.shape[0])])

    #Select K images randomly
    indices = np.random.choice(rgb.shape[0], K, replace=False)
    gray = np.stack([cv2.resize(gray_full[k], (RES,RES)) for k in range(rgb.shape[0])])
    gray = gray[indices]

    # Load UNLPS outputs
    # unle_dir = project_dir / "results" / model_name / "unle"
    # normals = np.load(unle_dir/"normals_unc.npy")
    # albedo  = np.load(unle_dir/"albedos_unc.npy")
    # depth   = np.load(unle_dir/"depth_unc.npy")

    unle_dir = project_dir / "results" / model_name / "gt"
    normals = np.load(unle_dir/"normals.npy")
    albedo  = np.load(unle_dir/"albedos.npy")
    depth   = np.load(unle_dir/"depth.npy")

    normals_r = np.stack([cv2.resize(normals[...,c], (RES,RES)) for c in range(3)], axis=-1)
    albedo_r  = cv2.resize(albedo, (RES,RES))
    depth_r   = cv2.resize(depth, (RES,RES))

    mask = (albedo_r > 0).astype(np.uint8)

    cam = dataset.metadata["camera"]
    intr = dict(fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"])

    X_unc = GeometryUtils.make_X_from_depth(depth_r, intr)
    B_unc = albedo_r[...,None] * normals_r

    # Convert GT light coords
    L_w = dataset.light_positions()
    L_w = L_w[indices]
    R = np.array(cam["R_cw"])[0] if np.array(cam["R_cw"]).ndim == 3 else np.array(cam["R_cw"])
    C = np.array(cam["location_world"])
    L_cam = (R @ (L_w - C).T).T

    L_test_w = test_dataset.light_positions()
    L_test_cam = (R @ (L_test_w - C).T).T
    L_test_cam = L_test_cam.reshape(-1, 3)

    L_cam = L_cam.reshape(-1, 3)
    L_test_cam = L_test_cam.reshape(-1, 3)

    # Build dataset + loaders
    ds = LightDatasetIAN(gray, albedo_r, normals_r, L_cam)
    loader_sup  = DataLoader(ds, batch_size=batch, shuffle=True)
    loader_phys = DataLoader(ds, batch_size=batch, shuffle=True)

    # Validation input (full batch)
    X_val = torch.from_numpy(np.stack([
        np.stack([gray[k]/gray[k].max(), albedo_r,
                  normals_r[...,0], normals_r[...,1], normals_r[...,2]], 0)
        for k in range(gray.shape[0])
    ])).float()

    X_test = torch.from_numpy(np.stack([
        np.stack([test_gray[k]/test_gray[k].max(), albedo_r,
                  normals_r[...,0], normals_r[...,1], normals_r[...,2]], 0)
        for k in range(test_gray.shape[0])
    ])).float()


    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = LightNet(in_channels=5).to(device)

    if resume:
        model.load_state_dict(torch.load(out_dir/"lightnet.pth"))
    else:
        model = LightNet(in_channels=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_unc).float().to(device)
    B_t = torch.from_numpy(B_unc).float().to(device)
    mask_t = torch.from_numpy(mask).bool().to(device)
    X_val = X_val.to(device)
    X_test = X_test.to(device)

    # Logging arrays
    sup_curve = []
    phys_curve = []
    total_curve = []
    val_eu = []
    test_eu = []


    # PHASE 1: supervised only
    print("\n=== PHASE 1: Supervised ===")
    for epoch in range(epochs1):
        sup, phys, total = train_one_epoch(
            model, loader_sup, loader_phys, optimizer, loss_fn,
            X_t, B_t, mask_t, lambda_phys=0.0, q=q, device=device
        )

        sup_curve.append(sup)
        phys_curve.append(phys)
        total_curve.append(total)

        ve = validate(model, X_val, L_cam, device)
        val_eu.append(ve)

        ve_test = validate(model, X_test, L_test_cam, device)
        test_eu.append(ve_test)

        print(f"[P1 {epoch+1}/{epochs1}] Sup={sup:.4f} | Val-Eu={ve:.4f} | Test-Eu={ve_test:.4f}")

    # PHASE 2: supervised + physics
    print("\n=== PHASE 2: Supervised + Physics ===")
    for epoch in range(epochs2):
        sup, phys, total = train_one_epoch(
            model, loader_sup, loader_phys, optimizer, loss_fn,
            X_t, B_t, mask_t, lambda_phys = min(0.5, epoch * 0.01), q=q, device=device
        )

        sup_curve.append(sup)
        phys_curve.append(phys)
        total_curve.append(total)

        ve = validate(model, X_val, L_cam, device)
        val_eu.append(ve)

        print(f"[P2 {epoch+1}/{epochs2}] Total={total:.4f} | Phys={phys:.5f} | Val-Eu={ve:.4f}")

        ve_test = validate(model, X_test, L_test_cam, device)
        test_eu.append(ve_test)
        print(f"[P2 {epoch+1}/{epochs2}] Total={total:.4f} | Phys={phys:.5f} | Test-Eu={ve_test:.4f}")

    plt.figure(figsize=(7,5))
    plt.plot(sup_curve, label="Supervised Loss")
    plt.plot(phys_curve, label="Physics Loss")
    plt.plot(total_curve, label="Total Loss")
    plt.plot(val_eu, label="Validation Euclidean Error")
    plt.plot(test_eu, label="Test Euclidean Error")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Validation and Test Error Curves")
    plt.tight_layout()
    plt.savefig(out_dir/"error_curve.png")
    plt.close()

    final_eu = float(val_eu[-1])
    final_eu_test = float(test_eu[-1])
    torch.save(model.state_dict(), out_dir/"lightnet.pth")

    metrics = {
        "final_euclidean": final_eu,
        "final_euclidean_test": final_eu_test,
        "sup_curve": sup_curve,
        "phys_curve": phys_curve,
        "total_curve": total_curve,
        "val_eu_curve": val_eu,
        "test_eu_curve": test_eu
    }
    json.dump(metrics, open(out_dir/"metrics.json", "w"), indent=2)

    print(" Experiment Completed")
    print(" Saved model + plots →", out_dir)

    #save plot comparing final light positions and directions with ground truth

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="bunny")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs1", type=int, default=200)
    p.add_argument("--epochs2", type=int, default=80)
    p.add_argument("--lambda_phys", type=float, default=0.1)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--q", type=int, default=3)
    # NEW — resume training option
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (.pth) to resume training from.")


    args = p.parse_args()

    run_experiment(
        model_name=args.model,
        lr=args.lr,
        batch=args.batch,
        epochs1=args.epochs1,
        epochs2=args.epochs2,
        lambda_phys=args.lambda_phys,
        q=args.q,
        K=args.K,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
