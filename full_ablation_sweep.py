#!/usr/bin/env python3
"""
Full parameter sweep ablation study for photometric stereo.
Tests actual algorithm variations with different hyperparameters.
"""

import sys
from pathlib import Path
import numpy as np
import json
import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))


def load_existing_data():
    """Load existing results for ground truth comparison."""
    project_dir = Path(__file__).parent
    results = {}
    
    for model in ['bunny', 'sphere']:
        model_dir = project_dir / "results" / model
        if not model_dir.exists():
            continue
            
        gt_dir = model_dir / "gt"
        if gt_dir.exists():
            results[model] = {
                'gt_normals': np.load(gt_dir / "normals.npy"),
                'gt_albedos': np.load(gt_dir / "albedos.npy"),
            }
            if (gt_dir / "light_directions.npy").exists():
                results[model]['gt_light_directions'] = np.load(gt_dir / "light_directions.npy")
    
    return results


def compute_metrics(pred_normals, pred_albedos, gt_normals, gt_albedos, mask=None):
    """Compute evaluation metrics."""
    if mask is None:
        gt_norms = np.linalg.norm(gt_normals, axis=2)
        mask = gt_norms > 0.5
    
    # Normal error
    normals_pred_flat = pred_normals[mask]
    normals_gt_flat = gt_normals[mask]
    
    # Normalize
    normals_pred_flat = normals_pred_flat / (
        np.linalg.norm(normals_pred_flat, axis=1, keepdims=True) + 1e-6
    )
    normals_gt_flat = normals_gt_flat / (
        np.linalg.norm(normals_gt_flat, axis=1, keepdims=True) + 1e-6
    )
    
    # Angular error
    dot_product = np.sum(normals_pred_flat * normals_gt_flat, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angular_error = np.arccos(dot_product) * 180.0 / np.pi
    
    # Albedo error
    albedo_pred_flat = pred_albedos[mask]
    albedo_gt_flat = gt_albedos[mask]
    
    return {
        'normal_error_mean': float(np.mean(angular_error)),
        'normal_error_median': float(np.median(angular_error)),
        'normal_error_std': float(np.std(angular_error)),
        'normal_error_95th': float(np.percentile(angular_error, 95)),
        'albedo_mae': float(np.mean(np.abs(albedo_pred_flat - albedo_gt_flat))),
        'albedo_rmse': float(np.sqrt(np.mean((albedo_pred_flat - albedo_gt_flat) ** 2))),
    }


def subsample_lights(images, light_directions, num_lights):
    """Subsample to use only specified number of lights."""
    if num_lights >= len(images):
        return images, light_directions
    
    # Evenly spaced sampling
    indices = np.linspace(0, len(images) - 1, num_lights, dtype=int)
    return images[indices], light_directions[indices]


def downsample_images(images, target_res):
    """Downsample images to target resolution."""
    n, h, w, c = images.shape
    if h == target_res and w == target_res:
        return images
    
    # Simple binning/averaging for downsampling
    factor_h = h // target_res
    factor_w = w // target_res
    
    downsampled = np.zeros((n, target_res, target_res, c), dtype=images.dtype)
    
    for i in range(n):
        for ch in range(c):
            for y in range(target_res):
                for x in range(target_res):
                    y_start = y * factor_h
                    x_start = x * factor_w
                    downsampled[i, y, x, ch] = np.mean(
                        images[i, y_start:y_start+factor_h, x_start:x_start+factor_w, ch]
                    )
    
    return downsampled


def uncalibrated_photometric_stereo_configurable(
    images: np.ndarray,
    num_lights: int,
    rank: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Uncalibrated photometric stereo with configurable rank.
    """
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError(f"Expected images shape (n, h, w, 3), got {images.shape}")

    n, h, w, _ = images.shape
    assert n == num_lights, "Number of images must match num_lights"

    # Convert to grayscale
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=images.dtype)
    I = np.tensordot(images, weights, axes=([3], [0]))
    I = I.reshape(n, h * w).T
    valid_mask_flat = (I.max(axis=1) > 0.01).astype(bool)
    I_valid = I[valid_mask_flat, :]

    # SVD factorization
    from scipy.linalg import svd
    U, S, Vt = svd(I_valid, full_matrices=False)

    # Use specified rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    M = U_r @ np.diag(np.sqrt(S_r))
    L_est = (np.diag(np.sqrt(S_r)) @ Vt_r).T

    # Normalize normals
    normals_valid = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-6)

    # Estimate albedos
    L_n = L_est @ normals_valid.T
    albedos_valid = np.zeros(I_valid.shape[0])
    for i in range(I_valid.shape[0]):
        denom = np.sum(L_n[:, i] ** 2) + 1e-6
        albedos_valid[i] = np.sum(I_valid[i, :] * L_n[:, i]) / denom
        albedos_valid[i] = max(0, albedos_valid[i])

    # Normalize lights
    light_norms = np.linalg.norm(L_est, axis=1)
    L_normalized = L_est / (light_norms[:, np.newaxis] + 1e-6)
    mean_light_norm = light_norms.mean()
    albedos_valid *= mean_light_norm

    # Coordinate transform
    image_to_camera = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=normals_valid.dtype)
    normals_valid = normals_valid @ image_to_camera.T
    L_normalized = L_normalized @ image_to_camera.T

    # Remap to full image
    normals_full = np.zeros((h * w, 3))
    albedos_full = np.zeros(h * w)
    normals_full[valid_mask_flat, :] = normals_valid
    albedos_full[valid_mask_flat] = albedos_valid
    normals_full[~valid_mask_flat, :] = np.array([0, 0, 1])
    albedos_full[~valid_mask_flat] = 0

    normals = normals_full.reshape(h, w, 3)
    albedos = albedos_full.reshape(h, w)
    valid_mask = valid_mask_flat.reshape(h, w)

    # Normalize
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.where(norms > 1e-6, normals / norms, normals)
    albedos = np.clip(albedos, 0, 1)

    return normals, albedos, L_normalized, valid_mask


def calibrated_photometric_stereo_configurable(
    images: np.ndarray,
    light_directions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrated photometric stereo.
    """
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError(f"Expected images shape (n, h, w, 3), got {images.shape}")
    if light_directions.ndim != 2 or light_directions.shape[1] != 3:
        raise ValueError(f"Expected light_directions shape (n, 3), got {light_directions.shape}")

    n, h, w, _ = images.shape
    assert n == light_directions.shape[0], "Number of images must match lights"

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=images.dtype)
    I = np.tensordot(images, weights, axes=([3], [0]))
    I = I.reshape(n, h * w).T

    valid_mask_flat = (I.max(axis=1) > 0.01).astype(bool)
    I_valid = I[valid_mask_flat, :]

    L = light_directions
    M_valid = np.linalg.lstsq(L, I_valid.T, rcond=None)[0].T

    albedos_valid = np.linalg.norm(M_valid, axis=1)
    normals_valid = M_valid / (albedos_valid[:, np.newaxis] + 1e-6)

    normals_full = np.zeros((h * w, 3))
    albedos_full = np.zeros(h * w)
    normals_full[valid_mask_flat, :] = normals_valid
    albedos_full[valid_mask_flat] = albedos_valid
    normals_full[~valid_mask_flat, :] = np.array([0, 0, 1])
    albedos_full[~valid_mask_flat] = 0

    normals = normals_full.reshape(h, w, 3)
    albedos = albedos_full.reshape(h, w)
    valid_mask = valid_mask_flat.reshape(h, w)

    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.where(norms > 1e-6, normals / norms, normals)
    albedos = np.clip(albedos, 0, 1)

    return normals, albedos, valid_mask


def run_ablation_sweep(model_name: str, method: str, output_dir: Path):
    """Run full parameter sweep ablation study."""
    print(f"\n{'='*70}")
    print(f"FULL ABLATION STUDY: {model_name.upper()} - {method.upper()}")
    print(f"{'='*70}\n")
    
    # Load ground truth
    gt_data = load_existing_data()
    if model_name not in gt_data:
        print(f"No ground truth data for {model_name}")
        return []
    
    gt_normals = gt_data[model_name]['gt_normals']
    gt_albedos = gt_data[model_name]['gt_albedos']
    
    # Try to load light directions
    gt_light_dirs = gt_data[model_name].get('gt_light_directions')
    if gt_light_dirs is None:
        print(f"Warning: No ground truth light directions for {model_name}")
        # Create dummy uniform hemisphere sampling
        n_lights = 30
        gt_light_dirs = np.array([[np.sin(theta) * np.cos(phi), 
                                   np.sin(theta) * np.sin(phi), 
                                   np.cos(theta)]
                                  for theta in np.linspace(0, np.pi/3, int(np.sqrt(n_lights)))
                                  for phi in np.linspace(0, 2*np.pi, int(np.sqrt(n_lights)))])[:n_lights]
    
    # Generate synthetic images from ground truth (simplified)
    # In real scenario, would load actual rendered images
    h, w = gt_normals.shape[:2]
    n_lights_max = len(gt_light_dirs)
    
    # Create synthetic intensities: I = albedo * max(0, N·L)
    synthetic_images = []
    for light_dir in gt_light_dirs:
        intensity = gt_albedos[:, :, np.newaxis] * np.maximum(0, 
            np.sum(gt_normals * light_dir[np.newaxis, np.newaxis, :], axis=2, keepdims=True))
        # Convert to RGB (grayscale)
        rgb_image = np.repeat(intensity, 3, axis=2)
        synthetic_images.append(rgb_image)
    synthetic_images = np.array(synthetic_images)
    
    # Define parameter grid
    if method == "uncalibrated_ff":
        param_grid = {
            'num_lights': [10, 15, 20, 25, 30],
            'image_res': [128, 256, 384, 512] if h >= 512 else [128, 256],
            'rank': [2, 3, 4, 5],
        }
    else:  # calibrated_ff
        param_grid = {
            'num_lights': [10, 15, 20, 25, 30],
            'image_res': [128, 256, 384, 512] if h >= 512 else [128, 256],
        }
    
    # Run sweep
    results = []
    config_count = 0
    
    # Count total configurations
    import itertools
    if method == "uncalibrated_ff":
        total_configs = len(param_grid['num_lights']) * len(param_grid['image_res']) * len(param_grid['rank'])
    else:
        total_configs = len(param_grid['num_lights']) * len(param_grid['image_res'])
    
    print(f"Total configurations to test: {total_configs}\n")
    
    for num_lights in param_grid['num_lights']:
        for image_res in param_grid['image_res']:
            if method == "uncalibrated_ff":
                rank_values = param_grid['rank']
            else:
                rank_values = [3]  # dummy for calibrated
            
            for rank in rank_values:
                config_count += 1
                
                if method == "uncalibrated_ff":
                    config = {'num_lights': num_lights, 'image_res': image_res, 'rank': rank}
                else:
                    config = {'num_lights': num_lights, 'image_res': image_res}
                
                print(f"[{config_count}/{total_configs}] Testing: {config}")
                
                start_time = time.time()
                
                try:
                    # Prepare data
                    images_subset, lights_subset = subsample_lights(
                        synthetic_images, gt_light_dirs, num_lights
                    )
                    
                    if image_res != h:
                        images_subset = downsample_images(images_subset, image_res)
                        gt_normals_ds = downsample_images(
                            gt_normals[np.newaxis, :, :, :], image_res
                        )[0]
                        gt_albedos_ds = downsample_images(
                            gt_albedos[np.newaxis, :, :, np.newaxis], image_res
                        )[0, :, :, 0]
                    else:
                        gt_normals_ds = gt_normals
                        gt_albedos_ds = gt_albedos
                    
                    # Run algorithm
                    if method == "uncalibrated_ff":
                        normals_pred, albedos_pred, lights_est, mask = \
                            uncalibrated_photometric_stereo_configurable(
                                images_subset, num_lights, rank=rank
                            )
                    else:  # calibrated_ff
                        normals_pred, albedos_pred, mask = \
                            calibrated_photometric_stereo_configurable(
                                images_subset, lights_subset
                            )
                    
                    # Compute metrics
                    metrics = compute_metrics(
                        normals_pred, albedos_pred,
                        gt_normals_ds, gt_albedos_ds,
                        mask
                    )
                    
                    elapsed_time = time.time() - start_time
                    metrics['time_seconds'] = elapsed_time
                    metrics['config'] = config
                    metrics['status'] = 'success'
                    
                    print(f"  ✓ Normal Error: {metrics['normal_error_mean']:.2f}° ± {metrics['normal_error_std']:.2f}°")
                    print(f"  ✓ Albedo MAE: {metrics['albedo_mae']:.4f}")
                    print(f"  ✓ Time: {elapsed_time:.2f}s\n")
                    
                except Exception as e:
                    print(f"  ✗ ERROR: {str(e)}\n")
                    metrics = {
                        'config': config,
                        'status': 'failed',
                        'error': str(e),
                        'time_seconds': time.time() - start_time,
                    }
                
                results.append(metrics)
    
    # Save results
    output_file = output_dir / f"ablation_full_{model_name}_{method}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"ABLATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}\n")
    
    # Sort by normal error
    successful_results = [r for r in results if r.get('status') == 'success']
    if successful_results:
        successful_results.sort(key=lambda x: x['normal_error_mean'])
        
        print("TOP 5 CONFIGURATIONS:")
        print("-" * 70)
        for i, r in enumerate(successful_results[:5]):
            print(f"{i+1}. {r['config']}")
            print(f"   Normal: {r['normal_error_mean']:.2f}° ± {r['normal_error_std']:.2f}°")
            print(f"   Albedo MAE: {r['albedo_mae']:.4f}, Time: {r['time_seconds']:.2f}s\n")
    
    return results


def main():
    """Run full ablation study."""
    print("\n" + "="*70)
    print("FULL PARAMETER SWEEP ABLATION STUDY")
    print("="*70)
    
    output_dir = Path(__file__).parent / "tuning_results"
    output_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    # Run for each model and method
    for model in ['bunny']:  # Can add 'sphere' if data exists
        for method in ['calibrated_ff', 'uncalibrated_ff']:
            results = run_ablation_sweep(model, method, output_dir)
            if results:
                if model not in all_results:
                    all_results[model] = {}
                all_results[model][method] = results
    
    # Find overall best
    best_overall = None
    best_error = float('inf')
    
    for model, methods in all_results.items():
        for method, results in methods.items():
            for r in results:
                if r.get('status') == 'success':
                    if r['normal_error_mean'] < best_error:
                        best_error = r['normal_error_mean']
                        best_overall = {
                            'model': model,
                            'method': method,
                            **r
                        }
    
    if best_overall:
        print("\n" + "="*70)
        print("BEST OVERALL CONFIGURATION")
        print("="*70)
        print(f"Model:         {best_overall['model']}")
        print(f"Method:        {best_overall['method']}")
        print(f"Config:        {best_overall['config']}")
        print(f"Normal Error:  {best_overall['normal_error_mean']:.2f}° ± {best_overall['normal_error_std']:.2f}°")
        print(f"Albedo MAE:    {best_overall['albedo_mae']:.4f}")
        print(f"Time:          {best_overall['time_seconds']:.2f}s")
        print("="*70 + "\n")
        
        # Save best config
        with open(output_dir / "best_config_full_sweep.json", 'w') as f:
            json.dump(best_overall, f, indent=2)


if __name__ == "__main__":
    main()
