#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import cv2

from UNLE.lightnet import LightNet
from UNLE.geometry_util import GeometryUtils
from util import load_dataset


###############################################################################
# Utility: physics-based reprojection error (Eq. 7)
###############################################################################
def physics_error(I_k, X, B, mask, L, q=2):
    """
    Compute physics reprojection error for a single light:

        I_pk ≈ e_k * B_p^T (L_k - X_p) / ||L_k - X_p||^q

    with e_k solved in closed form, then mean squared residual.
    All inputs must have consistent resolution H×W.
    """

    H, W = I_k.shape
    assert X.shape[:2] == (H, W), f"X shape {X.shape} vs I_k {I_k.shape}"
    assert B.shape[:2] == (H, W), f"B shape {B.shape} vs I_k {I_k.shape}"
    assert mask.shape == (H, W), f"mask shape {mask.shape} vs I_k {I_k.shape}"

    mask_flat = mask.ravel()
    Xf = X.reshape(-1, 3)[mask_flat]
    Bf = B.reshape(-1, 3)[mask_flat]
    If = I_k.ravel()[mask_flat]

    v = L[None, :] - Xf          # (N,3)
    r = np.linalg.norm(v, axis=1)
    r = np.maximum(r, 1e-8)
    phi = np.sum(Bf * v, axis=1) / (r ** q)  # (N,)

    denom = np.sum(phi * phi) + 1e-8
    ek = np.sum(If * phi) / denom

    residual = If - ek * phi
    return float(np.mean(residual ** 2))


###############################################################################
# Utility: angular error between predicted and GT lights
###############################################################################
def angular_error(pred, gt):
    pred_n = pred / (np.linalg.norm(pred) + 1e-8)
    gt_n = gt / (np.linalg.norm(gt) + 1e-8)
    dot = np.clip(np.dot(pred_n, gt_n), -1.0, 1.0)
    return float(np.arccos(dot)) * 180.0 / np.pi


###############################################################################
# Main testing script
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LightNet on dataset")
    parser.add_argument("--model", type=str, default="sphere")
    parser.add_argument(
        "--weights",
        type=str,
        default="lightnet_ian_physics.pth",
        help="Filename of trained LightNet weights (inside results/<model>/unle/)",
    )
    args = parser.parse_args()
    # hard override if you want:
    # args.model = "sphere"

    # -------------------------------------------------------------------------
    # Paths and dataset
    # -------------------------------------------------------------------------
    script_path = Path(__file__).resolve()
    project_dir = script_path.parent.parent
    dataset_dir = project_dir / "dataset" / "test" /args.model
    results_base = project_dir / "results" / args.model / "unle"

    print(f"Loading dataset at: {dataset_dir}")
    dataset = load_dataset(str(dataset_dir))

    # RGB → grayscale (original resolution, likely 512×512)
    rgb_images = dataset.rgb()          # (K,H_orig,W_orig,3)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    images_gray_full = np.tensordot(rgb_images, weights, axes=([3], [0])).astype(np.float32)
    K, H_full, W_full = images_gray_full.shape
    print(f"Original images_gray_full shape: {images_gray_full.shape}")

    # -------------------------------------------------------------------------
    # Load UNLPS results (normals, albedo, depth) — UNLPS resolution (H_u, W_u)
    # -------------------------------------------------------------------------
    print(f"Loading UNLPS results from: {results_base}")
    N_unc = np.load(results_base / "normals_unc.npy")     # (H_u,W_u,3)
    A_unc = np.load(results_base / "albedos_unc.npy")     # (H_u,W_u)
    
    z_unc = np.load(results_base / "depth_unc.npy")

    RES = 128
    K, Hf, Wf = images_gray_full.shape
    images_gray = np.zeros((K, RES, RES), np.float32)
    for k in range(K):
        images_gray[k] = cv2.resize(
            images_gray_full[k].astype(np.float32),
            (RES, RES),
            interpolation=cv2.INTER_AREA,
        )

    # Resize UNLPS outputs to 128x128 if needed
    if N_unc.shape[0] != RES or N_unc.shape[1] != RES:
        normals_res = np.zeros((RES, RES, 3), np.float32)
        for c in range(3):
            normals_res[..., c] = cv2.resize(
                N_unc[..., c], (RES, RES), interpolation=cv2.INTER_CUBIC
            )
        N_unc = normals_res

    if A_unc.shape != (RES, RES):
        A_unc = cv2.resize(
            A_unc.astype(np.float32),
            (RES, RES),
            interpolation=cv2.INTER_AREA,
        )

    if z_unc.shape != (RES, RES):
        z_unc = cv2.resize(
            z_unc.astype(np.float32),
            (RES, RES),
            interpolation=cv2.INTER_CUBIC,
        )

    
    H_u, W_u = A_unc.shape
    print(f"UNLPS resolution: {(H_u, W_u)}")

    # Define mask at UNLPS resolution
    mask = (A_unc > 0)


    # -------------------------------------------------------------------------
    # Construct X from depth (UNLPS resolution)
    # -------------------------------------------------------------------------
    cam = dataset.metadata["camera"]
    intr = {
        "fx": cam["fx"],
        "fy": cam["fy"],
        "cx": cam["cx"],
        "cy": cam["cy"],
    }
    X_unc = GeometryUtils.make_X_from_depth(z_unc, intr)  # (H_u,W_u,3)
    print(f"X_unc shape: {X_unc.shape}")

    # B = albedo * normal, used for physics reprojection
    B_unc = A_unc[:, :, None] * N_unc
    print(f"B_unc shape: {B_unc.shape}")

    # -------------------------------------------------------------------------
    # Convert GT light positions to camera coordinates
    # -------------------------------------------------------------------------
    light_positions_world = dataset.light_positions()          # (K,3)
    R_cw = np.array(cam["R_cw"])                               # world→cam
    C = np.array(cam["location_world"])
    # If R_cw is shape (3,3) just use it; if (1,3,3) take [0]
    if R_cw.ndim == 3:
        R_cw_use = R_cw[0]
    else:
        R_cw_use = R_cw
    L_gt_cam = (R_cw_use @ (light_positions_world - C).T).T     # (K,3)

    # -------------------------------------------------------------------------
    # Load trained model
    # -------------------------------------------------------------------------
    # Check if weights is a full path or relative to results_base
    if Path(args.weights).is_absolute():
        weights_path = Path(args.weights)
    elif (results_base / args.weights).exists():
        weights_path = results_base / args.weights
    elif (project_dir / args.weights).exists():
        weights_path = project_dir / args.weights
    else:
        weights_path = Path(args.weights)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    print(f"\nLoading LightNet weights from: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightNet(in_channels=5).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # -------------------------------------------------------------------------
    # Build test set: (I, A, N) packed as 5 channels at UNLPS resolution
    # -------------------------------------------------------------------------
    X_test_list = []
    for k in range(K):
        I_k = images_gray[k]
        if I_k.max() > 0:
            I_k = I_k / (I_k.max() + 1e-8)

        x = np.stack(
            [
                I_k,
                A_unc,
                N_unc[:, :, 0],
                N_unc[:, :, 1],
                N_unc[:, :, 2],
            ],
            axis=0,
        )  # (5,H_u,W_u)

        X_test_list.append(x)

    X_test = torch.from_numpy(np.stack(X_test_list, axis=0)).float().to(device)  # (K,5,H_u,W_u)

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    with torch.no_grad():
        L_pred = model(X_test)      # (K,3)
    L_pred_np = L_pred.cpu().numpy()

    print("\n=== Evaluation Metrics ===")

    euclid_errs = []
    angle_errs = []
    phys_errs = []

    q = 2  # near-light falloff exponent you used in UNLPS

    for k in range(K):
        pred = L_pred_np[k]
        gt = L_gt_cam[k]

        euclid = np.linalg.norm(pred - gt)
        angle = angular_error(pred, gt)
        phys = physics_error(images_gray[k], X_unc, B_unc, mask, pred, q=q)

        euclid_errs.append(euclid)
        angle_errs.append(angle)
        phys_errs.append(phys)

        # print (f"Light gt: {gt}")
        # print (f"Light pred: {pred}")

        # print(
        #     f"Light {k:02d}:  "
        #     f"Euclid={euclid:.4f},  "
        #     f"Angle={angle:.3f} deg,  "
        #     f"PhysicsErr={phys:.4e}"
        # )

    print("\nMean Euclidean Error:", np.mean(euclid_errs))
    print("Mean Angular Error (deg):", np.mean(angle_errs))
    print("Mean Physics Reprojection Error:", np.mean(phys_errs))

    #save predictions and ground truth
    np.save(results_base / "predictions.npy", L_pred_np)
    np.save(results_base / "ground_truth.npy", L_gt_cam)
    np.save(results_base / "euclid_errs.npy", euclid_errs)
    np.save(results_base / "angle_errs.npy", angle_errs)
    np.save(results_base / "phys_errs.npy", phys_errs)

    # ax.scatter(
    #     L_pred_np[:, 0],
    #     L_pred_np[:, 1],
    #     L_pred_np[:, 2],
    #     c="red",
    #     s=50,
    #     label="Predicted",
    # )

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.legend()
    # ax.set_box_aspect([1, 1, 1])
    # plt.title("Predicted vs Ground Truth Light Positions (Camera Frame)")
    # plt.show()


if __name__ == "__main__":
    main()
