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

    def gradient_descent_refine(self, I_k, X, B, mask,
                                Lk_init, ek_init,
                                num_steps=30, lr=0.02):
        """
        Refine light position and intensity using gradient descent,
        following the gradient expression in the paper.

        Args:
            I_k: (H, W) image for light k
            X: (H, W, 3) 3D points
            B: (H, W, 3) scaled normals
            mask: (H, W) valid pixel mask
            Lk_init: (3,) initial light position
            ek_init: float, initial light intensity
            num_steps: number of gradient descent iterations
            lr: learning rate

        Returns:
            Lk: (3,) refined light position
            ek: refined intensity
        """
        Lk = Lk_init.astype(np.float32).copy()
        ek = float(ek_init)

        mask_flat = mask.ravel()
        Xf = X.reshape(-1, 3)[mask_flat]    # (N_valid, 3)
        Bf = B.reshape(-1, 3)[mask_flat]    # (N_valid, 3)
        If = I_k.ravel()[mask_flat]         # (N_valid,)

        N_valid = len(Xf)
        if N_valid == 0:
            return Lk, ek

        for step in range(num_steps):
            # v_p = Lk - Xp
            v = Lk[None, :] - Xf           # (N_valid, 3)
            r = norm(v, axis=1, keepdims=True) + 1e-8   # (N_valid,1)
            r_pow_q = r ** self.q
            r_pow_q_plus_2 = r ** (self.q + 2)

            # phi_p = B_p^T (Lk - Xp) / ||Lk - Xp||^q
            B_dot_v = np.sum(Bf * v, axis=1, keepdims=True)       # (N_valid,1)
            phi = B_dot_v / r_pow_q                               # (N_valid,1)

            # residual_p = I_p - e_k * phi_p
            residual = If[:, None] - ek * phi                     # (N_valid,1)

            # term from ∂F/∂Lk (vectorized version of paper's eq. 9)
            term = Bf * (r ** 2) - self.q * v * B_dot_v           # (N_valid,3)

            # gradient is sum over pixels
            gradient = -2.0 * ek * np.sum(
                residual * term / r_pow_q_plus_2,
                axis=0
            )  # (3,)

            # gradient descent update
            Lk_new = Lk - lr * gradient

            # Prevent insane jumps
            if np.linalg.norm(Lk_new) <= 10.0 and np.isfinite(Lk_new).all():
                Lk = Lk_new

            # Re-estimate ek in closed form given current Lk
            v_new = Lk[None, :] - Xf
            r_new = norm(v_new, axis=1, keepdims=True) + 1e-8
            phi_new = np.sum(Bf * v_new, axis=1, keepdims=True) / (r_new ** self.q)

            denom = np.sum(phi_new * phi_new) + 1e-8
            ek = float(np.sum(If[:, None] * phi_new) / denom)
            ek = np.clip(ek, 0.0, 100.0)

            # Optional early stopping
            if step % 10 == 0 and norm(gradient) < 1e-4:
                break

        return Lk, ek

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
