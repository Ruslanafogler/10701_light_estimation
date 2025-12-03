"""
Light position and intensity estimation for uncalibrated photometric stereo.
"""

import numpy as np
from numpy.linalg import norm, solve


class LightEstimator:
    """
        I_pk ≈ e_k * B_p^T (L_k - X_p) / ||L_k - X_p||^q
    """

    def __init__(self, q=2):
        self.q = q

    def gradient_descent_refine(
        self, I_k, X, B, mask, Lk_init, ek_init,
        num_steps=200, lr=0.01, tol=1e-4):
        """
        Gradient descent refinement for light position L_k, based on Eq. (9)
        in Papadhimitri & Favaro (UNLPS), keeping e_k fixed during GD and
        re-estimating it once at the end.

        Args:
            I_k:   (H, W) grayscale image for light k
            X:     (H, W, 3) 3D points in camera coordinates
            B:     (H, W, 3) pseudonormals (albedo * normal)
            mask:  (H, W) boolean mask of valid pixels
            Lk_init: (3,) initial light position (camera coords)
            ek_init: scalar initial intensity
            num_steps: gradient descent iterations
            lr: learning rate
            tol: early-stop threshold on grad norm

        Returns:
            Lk: (3,) refined light position
            ek: scalar refined intensity
        """

        # Work in double precision for stability
        Lk = Lk_init.astype(np.float64).copy()
        ek = float(ek_init)

        mask_flat = mask.ravel()
        Xf = X.reshape(-1, 3)[mask_flat]   # (N, 3)
        Bf = B.reshape(-1, 3)[mask_flat]   # (N, 3)
        If = I_k.ravel()[mask_flat].astype(np.float64)  # (N,)

        N = len(Xf)
        if N == 0:
            return Lk_init, ek_init

        for step in range(num_steps):
            # v_p = L - X_p
            v = Lk[None, :] - Xf                 # (N, 3)
            r = np.linalg.norm(v, axis=1, keepdims=True)  # (N, 1)
            r = np.maximum(r, 1e-6)

            # B_p^T v_p
            B_dot_v = np.sum(Bf * v, axis=1, keepdims=True)  # (N, 1)

            # φ_p = B_p^T v_p / ||v_p||^q
            phi = B_dot_v / (r ** self.q)                    # (N, 1)

            # residual: I_p - e_k φ_p
            residual = If[:, None] - ek * phi                # (N, 1)

            # term1 = B_p / ||v_p||^q
            term1 = Bf / (r ** self.q)                       # (N, 3)

            # term2 = q * (B_p^T v_p) v_p / ||v_p||^{q+2}
            term2 = (self.q * B_dot_v * v) / (r ** (self.q + 2))

            # ∂F/∂L_k
            grad = -2.0 * ek * np.sum(
                residual * (term1 - term2),
                axis=0
            )  # (3,)

            gnorm = np.linalg.norm(grad)

            # Early stopping if gradient is tiny
            if gnorm < tol:
                # print(f"[refine] early stop at step {step}, |grad|={gnorm:.2e}")
                break

            # Gradient descent update
            step_vec = lr * grad

            # Optional: clamp max step to avoid jumps
            max_step = 0.05
            step_norm = np.linalg.norm(step_vec)
            if step_norm > max_step:
                step_vec *= (max_step / step_norm)

            Lk_new = Lk - step_vec

            # Simple sanity clamp on radius
            if np.isfinite(Lk_new).all() and np.linalg.norm(Lk_new) < 20.0:
                Lk = Lk_new

            # (Optional) monitor
            if step % 50 == 0:
                F = np.sum(residual**2)
                print(f"[refine] step {step}, F={F:.4e}, |grad|={gnorm:.2e}")

        # After optimizing L_k, recompute e_k in closed form (LS)
        v_final = Lk[None, :] - Xf
        r_final = np.linalg.norm(v_final, axis=1, keepdims=True)
        r_final = np.maximum(r_final, 1e-6)
        B_dot_v_final = np.sum(Bf * v_final, axis=1, keepdims=True)
        phi_final = B_dot_v_final / (r_final ** self.q)

        denom = np.sum(phi_final ** 2) + 1e-8
        ek = float(np.sum(If[:, None] * phi_final) / denom)
        ek = np.clip(ek, 0.0, 100.0)

        return Lk.astype(np.float32), ek


    def update_B(self, I, X, L, e, mask):
        """
        Solve for scaled normals B_p at each pixel:

            I_pk ≈ e_k * B_p^T (L_k - X_p) / ||L_k - X_p||^q

        Args:
            I: (K, H, W) images
            X: (H, W, 3) 3D points
            L: (K, 3) light positions
            e: (K,) intensities
            mask: (H, W) valid pixels

        Returns:
            B: (H, W, 3) scaled normals
        """
        K, H, W = I.shape
        B = np.zeros((H, W, 3), np.float32)

        mask_flat = mask.ravel()
        valid_idx = np.where(mask_flat)[0]
        if len(valid_idx) == 0:
            return B

        X_flat = X.reshape(-1, 3)
        I_flat = I.reshape(K, -1).T  # (H*W, K)

        X_valid = X_flat[valid_idx]   # (N_valid, 3)
        I_valid = I_flat[valid_idx]   # (N_valid, K)
        N_valid = len(valid_idx)

        # Vectorized geometry: v_p_k = L_k - X_p
        v = L[None, :, :] - X_valid[:, None, :]   # (N_valid, K, 3)
        r = norm(v, axis=2) + 1e-8                # (N_valid, K)
        rpow = r ** self.q                        # (N_valid, K)

        # b_p_k = e_k * (L_k - X_p) / ||L_k - X_p||^q
        b = (e[None, :, None] * v) / rpow[:, :, None]  # (N_valid, K, 3)

        B_valid = np.zeros((N_valid, 3), np.float32)

        # Solve b_p @ B_p = I_p for each pixel (3 unknowns, K equations)
        for i in range(N_valid):
            b_p = b[i]       # (K, 3)
            I_p = I_valid[i] # (K,)

            bTb = b_p.T @ b_p   # (3,3)
            bTI = b_p.T @ I_p   # (3,)

            try:
                Bp = solve(bTb, bTI)
                if not np.isfinite(Bp).all():
                    Bp = np.zeros(3, dtype=np.float32)
            except Exception:
                Bp = np.zeros(3, dtype=np.float32)

            B_valid[i] = Bp

        B_flat = B.reshape(-1, 3)
        B_flat[valid_idx] = B_valid
        return B_flat.reshape(H, W, 3)

    def coarse_search(self, I_k, X, B, mask, bounds, step):
        """
        Coarse search for a single light given current geometry X, pseudonormals B.
        """

        # valid pixels
        mask_flat = mask.ravel()
        Xf = X.reshape(-1, 3)[mask_flat]
        Bf = B.reshape(-1, 3)[mask_flat]
        If = I_k.ravel()[mask_flat].astype(np.float64)

        N = len(Xf)
        if N == 0:
            # degenerate
            return np.zeros(3, dtype=np.float32), 1.0

        # center the search around object center, not origin
        center = Xf.mean(axis=0)  # (3,)

        # local search cube around center (in camera coords)
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        xs = np.arange(xmin, xmax + 1e-6, step)
        ys = np.arange(ymin, ymax + 1e-6, step)
        zs = np.arange(zmin, zmax + 1e-6, step)

        best_F = np.inf
        best_L = None
        best_e = None

        for z in zs:
            for y in ys:
                for x in xs:
                    Lk = center + np.array([x, y, z], dtype=np.float64)

                    v = Lk[None, :] - Xf              # (N,3)
                    r = np.linalg.norm(v, axis=1) + 1e-8
                    rpow = r ** self.q

                    phi = np.sum(Bf * v, axis=1) / rpow  # (N,)

                    denom = np.sum(phi * phi) + 1e-8
                    e_k = np.sum(If * phi) / denom

                    residual = If - e_k * phi
                    F = np.sum(residual * residual)

                    if F < best_F:
                        best_F = F
                        best_L = Lk.copy()
                        best_e = e_k

        return best_L.astype(np.float32), float(best_e)