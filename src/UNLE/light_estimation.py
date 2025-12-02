"""
Light position and intensity estimation for uncalibrated photometric stereo.
"""

import numpy as np
from numpy.linalg import norm, solve


class LightEstimator:
    """
    Estimate light positions and intensities for the near-light Lambertian model:

        I_pk ≈ e_k * B_p^T (L_k - X_p) / ||L_k - X_p||^q
    """

    def __init__(self, q=2):
        """
        Args:
            q: inverse distance exponent (2 or 3, usually 2 for near-light)
        """
        self.q = q

    def gradient_descent_refine(
            self, I_k, X, B, mask, Lk_init, ek_init,
            num_steps=1000, lr=0.001):
        """
        Gradient descent refinement for light position, based on Eq. (9) in UNLPS.
        """

        Lk = Lk_init.astype(np.float64).copy()
        ek = float(ek_init)

        mask_flat = mask.ravel()
        Xf = X.reshape(-1, 3)[mask_flat]
        Bf = B.reshape(-1, 3)[mask_flat]

        # Normalize B so it behaves like normals
        Bn = Bf / (np.linalg.norm(Bf, axis=1, keepdims=True) + 1e-8)

        If = I_k.ravel()[mask_flat]
        N = len(Xf)
        if N == 0:
            return Lk_init, ek_init

        for step in range(num_steps):
            # v_p = L - X_p
            v = Lk[None, :] - Xf                     # (N, 3)
            r = np.linalg.norm(v, axis=1, keepdims=True)
            r = np.maximum(r, 1e-6)

            # φ_p = B_p^T v_p / ||v_p||^q
            B_dot_v = np.sum(Bn * v, axis=1, keepdims=True)    # (N,1)
            phi     = B_dot_v / (r ** self.q)                  # (N,1)

            # residual: I_p - e_k φ_p
            residual = If[:, None] - ek * phi                  # (N,1)

            # term in ∂F/∂L (vectorized version of Eq. 9)
            term = Bn * (r ** 2) - self.q * v * B_dot_v        # (N,3)

            # gradient w.r.t L (no division by N!)
            grad = -2.0 * ek * np.sum(
                residual * term / (r ** (self.q + 2)),
                axis=0
            )                                                  # (3,)

            # OPTIONAL: scale step to fixed max length (stabilizer)

            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-9:
                step_vec = lr * grad / gnorm    # normalize step
            else:
                step_vec = 0

            Lk_new = Lk - step_vec

            # sanity clamp (just to avoid crazy explosions)
            if np.isfinite(Lk_new).all() and np.linalg.norm(Lk_new) < 20.0:
                Lk = Lk_new

            # Re-estimate ek in closed form
            v_new   = Lk[None, :] - Xf
            r_new   = np.linalg.norm(v_new, axis=1, keepdims=True)
            r_new   = np.maximum(r_new, 1e-6)
            phi_new = np.sum(Bn * v_new, axis=1, keepdims=True) / (r_new ** self.q)

            denom = np.sum(phi_new ** 2) + 1e-8
            ek    = float(np.sum(If[:, None] * phi_new) / denom)
            ek    = np.clip(ek, 0.0, 100.0)

            # Early stopping, but with a *gradient norm before scaling*
            if step % 20 == 0 and gnorm < 1e-3:
                # print(f"Early stopping at step {step}, grad norm={gnorm:.2e}")
                break

        return Lk.astype(np.float32), ek


    def update_B(self, I, X, L, e, mask):
        """
        Solve for scaled normals B_p at each pixel:

            I_pk ≈ e_k * B_p^T (L_k - X_p) / ||L_k - X_p||^q

        This is linear in B_p for fixed lights (per pixel least squares).

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
        Args:
            I_k : (H, W) image under light k
            X   : (H, W, 3) 3D points
            B   : (H, W, 3) scaled normals B_p
            mask: (H, W) boolean valid pixel mask
            bounds : ((xmin,xmax),(ymin,ymax),(zmin,zmax))
            step : grid step (e.g., 0.1 in paper)

        Returns:
            best_L : (3,) best light position
            best_e : scalar intensity
        """

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

        # Create 1D grids for search (paper used 10 cm steps)
        xs = np.arange(xmin, xmax + 1e-6, step)
        ys = np.arange(ymin, ymax + 1e-6, step)
        zs = np.arange(zmin, zmax + 1e-6, step)

        # Extract valid pixels (flattened)
        mask_flat = mask.ravel()
        Xf = X.reshape(-1, 3)[mask_flat]    # (N, 3)
        Bf = B.reshape(-1, 3)[mask_flat]    # (N, 3)
        If = I_k.ravel()[mask_flat]         # (N,)

        best_F = np.inf
        best_L = None
        best_e = None

        # Scan entire volume (paper: 1 m cube, 10 cm resolution)
        for z in zs:
            for y in ys:
                for x in xs:
                    Lk = np.array([x, y, z], dtype=np.float32)

                    # Eq(7) terms: v = L - Xp
                    v = Lk[None, :] - Xf         # (N, 3)
                    r = np.linalg.norm(v, axis=1) + 1e-8
                    rpow = r ** self.q

                    # φ_p = B_p^T (L - X_p) / ||L - X_p||^q
                    phi = np.sum(Bf * v, axis=1) / rpow   # (N,)

                    # Eq(8) closed form for light intensity e
                    denom = np.sum(phi * phi) + 1e-8
                    e_k = np.sum(If * phi) / denom

                    # Residual energy F(Lk)
                    residual = If - e_k * phi
                    F = np.sum(residual * residual)

                    # Keep best
                    if F < best_F:
                        best_F = F
                        best_L = Lk
                        best_e = e_k

        return best_L, best_e