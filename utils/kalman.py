import numpy as np
from utils.simulator import BALL_RADIUS, DEFAULT_CAMERA, DEFAULT_FIELD
import matplotlib.pyplot as plt
import utils.network as network
from tqdm import tqdm
import pandas as pd
import torch
import cv2
import utils.data as dt

CAMERA = DEFAULT_CAMERA
FIELD = DEFAULT_FIELD

# Values for the Kalman Filters
DELTA_T = 0.01
PROCESS_VAR_VEL_XY = 0.1
PROCESS_VAR_VEL_Z  = 0.1
MEAS_VAR_POS       = 0.5

# Values for the majority voting regime
Z_FLOOR = 0.1
MARGIN = 0.15 # margin for hysteresis
T_DOWN  = Z_FLOOR - MARGIN
T_UP    = Z_FLOOR + MARGIN

class CA3DKalmanFilter:
    def __init__(self, dt, process_var_acc, meas_var_pos = None,
                 initial_state=None, initial_covariance=1e3):
        """
        3D constant-acceleration Kalman Filter with state [x,y,z,vx,vy,vz,ax,ay,az].
        """
        self.dt = dt
        self.dim_x = 9
        self.dim_z = 3

        # State
        self.x = np.zeros((9,1))
        if initial_state is not None:
            self.x = initial_state.reshape((9,1))

        # Covariance
        self.P = np.eye(self.dim_x) * initial_covariance

        # Build F
        F = np.zeros((9,9))
        for i in range(3):
            F[i, i] = 1
            F[i, 3+i] = dt
            F[i, 6+i] = 0.5 * dt**2
            F[3+i, 3+i] = 1
            F[3+i, 6+i] = dt
            F[6+i, 6+i] = 1
        self.F = F

        # Measurement matrix H (positions only)
        H = np.zeros((3,9))
        H[0,0] = H[1,1] = H[2,2] = 1
        self.H = H

        # Process noise
        Q = np.zeros((9,9))
        Q[6,6] = process_var_acc
        Q[7,7] = process_var_acc
        Q[8,8] = process_var_acc
        self.Q = Q

        # Measurement noise
        if meas_var_pos is not None:
            self.R = np.eye(3) * meas_var_pos
        else:
            self.R = np.eye(3)

    def predict(self):
        # x = F x
        self.x = self.F @ self.x
        # P = F P F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        z: (3,) or (3,1) measurement vector [x,y,R]
        """
        z = z.reshape((3,1))
        radius = z[2,0]
        # Innovation
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # State update
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P


class CA3DKalmanFilter_ZeroXY:
    def __init__(self, dt, process_var_acc_z, meas_var_pos=None,
                 initial_state=None, initial_covariance=1e3):
        """
        3D Kalman Filter with:
          - state  = [x, y, z, vx, vy, vz, ax, ay, az]^T
          - but we FORCE ax = 0, ay = 0 every step.
          - only az is treated as a constant-acc state.

        Arguments:
          dt                : time step Δt
          process_var_acc_z : process‐noise variance for a_z only
          meas_var_pos      : if provided, sets R = I*meas_var_pos; else R = I
          initial_state     : 9×1 array to init x (if None, x=0)
          initial_covariance: scalar to init P = I * initial_covariance
        """
        self.dt = dt
        self.dim_x = 9
        self.dim_z = 3

        # 1) State vector x
        self.x = np.zeros((9,1))
        if initial_state is not None:
            # make sure it’s a column vector of length 9
            self.x = initial_state.reshape((9,1))

        # 2) Covariance P
        self.P = np.eye(self.dim_x) * initial_covariance

        # 3) Build new F that zeroes out ax, ay dynamics
        F = np.zeros((9,9))

        for axis in range(3):
            # row ‘axis’ handles position update (x,y,z)
            # originally: F[axis, axis] = 1
            #             F[axis, 3+axis] = dt
            #             F[axis, 6+axis] = 0.5*dt^2
            #
            # To force a_x=0 for axis=0 (x),
            #  we set F[0,6] -> 0.  Similarly for axis=1.
            #
            F[axis, axis] = 1
            F[axis, 3 + axis] = dt
            if axis == 2:
                # keep z’s acc term
                F[axis, 6 + axis] = 0.5 * dt**2
            else:
                # zero out x-acc or y-acc contribution
                F[axis, 6 + axis] = 0.0

            # row ‘3+axis’ handles velocity update (vx,vy,vz)
            F[3 + axis, 3 + axis] = 1
            if axis == 2:
                # only z-acc influences vz
                F[3 + axis, 6 + axis] = dt
            else:
                # force vx_{k+1} = vx_k, no a_x term
                F[3 + axis, 6 + axis] = 0.0

            # row ‘6+axis’ handles acceleration transition (ax,ay,az)
            if axis == 2:
                # keep a_z as a constant state
                F[6 + axis, 6 + axis] = 1
            else:
                # force a_x = 0, a_y = 0 at next step
                F[6 + axis, 6 + axis] = 0.0

        self.F = F

        # 4) Measurement matrix H (we still measure [x,y,z])
        H = np.zeros((3,9))
        H[0, 0] = 1   # measure x
        H[1, 1] = 1   # measure y
        H[2, 2] = 1   # measure z
        self.H = H

        # 5) Process‐noise Q
        # We only want process‐noise on a_z (index 8).
        Q = np.zeros((9,9))
        Q[8,8] = process_var_acc_z
        # a_x and a_y have zero process noise, so they stay pinned at zero.
        self.Q = Q

        # 6) Measurement noise R
        if meas_var_pos is not None:
            self.R = np.eye(3) * meas_var_pos
        else:
            self.R = np.eye(3)

    def predict(self):
        # x_{k+1} = F x_k
        self.x = self.F @ self.x

        # P_{k+1} = F P F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        # (Now a_x and a_y will be zero, because F forced them to zero)

    def update(self, z):
        """
        z : (3,) or (3,1) measurement [x, y, z].
        """
        z = z.reshape((3,1))

        # 1) Innovation
        y = z - (self.H @ self.x)

        # 2) Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # 3) Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 4) State update
        self.x = self.x + K @ y

        # 5) Covariance update
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        # If any small numerical round‐off introduced a nonzero a_x or a_y,
        # they will be reset to zero on the next predict(), because F forces them to zero.


class CA3DKF_withKnownAz:
    def __init__(self, dt, process_var_vel, meas_var_pos,
                 initial_state=None, initial_covariance=1e3):
        """
        A 6D constant‐velocity KF in 3D, but with a KNOWN vertical acceleration u_k.

        State vector (6×1): [x, y, z, vx, vy, vz]^T.
        Measurement (3×1):    [x_meas, y_meas, z_meas].

        dt               : time step (seconds or frames).
        process_var_vel  : variance on v_x, v_y, v_z (e.g. (px/frame)² or (m/sec)²).
        meas_var_pos     : variance on x,y,z measurements (e.g. (px)² or (m)²).
        initial_state    : optional 6×1 array to seed the filter (defaults to zeros).
        initial_covariance: scalar to initialize P = I * initial_covariance.
        """
        self.dt = dt
        self.dim_x = 6   # [x,y,z, vx,vy,vz]
        self.dim_z = 3   # [x,y,z] measurement

        # 1) State vector x (6×1)
        self.x = np.zeros((6,1))
        if initial_state is not None:
            self.x = initial_state.reshape((6,1))

        # 2) Covariance P (6×6)
        self.P = np.eye(self.dim_x) * initial_covariance

        # 3) F (6×6): constant‐velocity model (no accel in state)
        F = np.zeros((6,6))
        for i in range(3):
            F[i,   i]     = 1           # x_k+1 = x_k + vx_k * dt + ...
            F[i,   3 + i] = dt
            F[3+i, 3 + i] = 1           # vx_k+1 = vx_k (plus input for z only)
        self.F = F

        # 4) B (6×1) maps the KNOWN a_z into Δz, Δvz
        #    [Δx, Δy, Δz, Δvx, Δvy, Δvz]^T due to a_z = [0,0,u_k]^T
        self.B = np.array([
            [0.0],             # no direct ∆x from a_z
            [0.0],             # no direct ∆y from a_z
            [0.5 * dt**2],     # ∆z = ½ * a_z * dt²
            [0.0],             # no ∆vx from a_z
            [0.0],             # no ∆vy from a_z
            [dt]               # ∆vz = a_z * dt
        ])

        # 5) H (3×6): we measure only (x,y,z)
        H = np.zeros((3,6))
        H[0,0] = 1  # measure x
        H[1,1] = 1  # measure y
        H[2,2] = 1  # measure z
        self.H = H

        # 6) Q (6×6): process noise on velocities
        Q = np.zeros((6,6))
        Q[3,3] = process_var_vel   # variance on v_x
        Q[4,4] = process_var_vel   # variance on v_y
        Q[5,5] = process_var_vel   # variance on v_z
        self.Q = Q

        # 7) R (3×3): measurement noise on (x,y,z)
        self.R = np.eye(3) * meas_var_pos

    def predict(self, u=0.0):
        """
        Predict step, with known vertical acceleration u (scalar):
          x_{k+1} = F x_k + B u_k
          P_{k+1} = F P_k F^T + Q
        u = a_z (either 0 or -9.8)
        """
        # State‐predict: 
        self.x = (self.F @ self.x) + (self.B * u)

        # Covariance‐predict:
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Standard KF update with measurement z = [x_meas, y_meas, z_meas]^T.
        """
        z = z.reshape((3,1))
        # Innovation
        y = z - (self.H @ self.x)
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # State update
        self.x = self.x + (K @ y)
        # Covariance update
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

class ThrowKF:
    """
    6D Kalman Filter in 3D with KNOWN vertical acceleration a_z = -9.8.

    State = [x, y, z, vx, vy, vz]^T
    Measurement = [x, y, z]^T
    """

    def __init__(self, dt, process_var_vel, meas_var_pos, initial_cov=1e3):
        self.dt = dt
        self.dim_x = 6
        self.dim_z = 3

        # 1) State vector x (6×1)
        self.x = np.zeros((6,1))

        # 2) Covariance P (6×6)
        self.P = np.eye(self.dim_x) * initial_cov

        # 3) F (6×6): constant‐velocity block
        F = np.zeros((6,6))
        for i in range(3):
            F[i,   i]   = 1
            F[i,   3+i] = dt
            F[3+i, 3+i] = 1
        self.F = F

        # 4) B (6×1) maps a_z into Δz, Δv_z
        self.B = np.array([
            [0.0],
            [0.0],
            [0.5 * dt**2],
            [0.0],
            [0.0],
            [dt]
        ])

        # 5) H (3×6): measure (x,y,z)
        H = np.zeros((3,6))
        H[0,0] = 1
        H[1,1] = 1
        H[2,2] = 1
        self.H = H

        # 6) Q (6×6): process noise on vx, vy, vz
        Q = np.zeros((6,6))
        Q[3,3] = process_var_vel
        Q[4,4] = process_var_vel
        Q[5,5] = process_var_vel
        self.Q = Q

        # 7) R (3×3): measurement noise on (x,y,z)
        if isinstance(meas_var_pos, np.ndarray) and meas_var_pos.shape == (3, 3):
            self.R = meas_var_pos
        else:
            self.R = np.eye(3) * meas_var_pos

        # 8) We’ll always use u = -9.8  (gravity)
        self.u_const = -9.8

    def predict(self, u=None):
        """
        Predict step with known vertical acceleration u_k (default = -9.8).
        x_{k+1} = F x_k + B u_k,   P_{k+1} = F P F^T + Q
        """
        if u is None:
            u = self.u_const

        self.x = (self.F @ self.x) + (self.B * u)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Standard KF update with measurement z = [x_meas, y_meas, z_meas].
        """
        z = z.reshape((3,1))
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

class RollKF:
    """
    6D Kalman Filter (“rolling” model) with state = [x, y, z, vx, vy, vz]^T,
    but we FORCE vz = 0, az = 0.  z itself always comes from measurements.

    - (x,y) follow a constant‐velocity model.
    - z is “carried forward” between measurements (predict keeps z_k),
      but update() resets z to the new measurement.
    - vz is pinned to zero every step; az is implicitly zero because we never
      estimate it (no control input).
    """

    def __init__(self, dt, process_var_vel_xy, meas_var_pos, initial_cov=1e3):
        """
        Args:
          dt                : time step (seconds or frames).
          process_var_vel_xy: variance on v_x, v_y (e.g. (px/frame)² or (m/sec)²).
          meas_var_pos      : variance on x,y,z measurements (same units²).
          initial_cov       : scalar to initialize P = I * initial_cov (for all 6 dims).
        """
        self.dt = dt
        self.dim_x = 6   # [x, y, z, vx, vy, vz]
        self.dim_z = 3   # [x, y, z] measurement

        # 1) State vector x (6×1): [x, y, z, vx, vy, vz]
        self.x = np.zeros((6,1))

        # 2) Covariance P (6×6)
        self.P = np.eye(self.dim_x) * initial_cov

        # 3) F (6×6) for rolling, where:
        #    - x_{k+1} = x_k + vx_k * dt
        #    - y_{k+1} = y_k + vy_k * dt
        #    - z_{k+1} = z_k           (since vz->0)
        #    - vx_{k+1} = vx_k         (CV in x)
        #    - vy_{k+1} = vy_k         (CV in y)
        #    - vz_{k+1} = 0            (we pin vz to 0)
        F = np.zeros((6,6))
        # x‐row
        F[0,0] = 1
        F[0,3] = dt
        # y‐row
        F[1,1] = 1
        F[1,4] = dt
        # z‐row: z_{k+1} = z_k  (no dependence on vz)
        F[2,2] = 1
        # vx‐row
        F[3,3] = 1
        # vy‐row
        F[4,4] = 1
        # vz‐row: pinned to zero -> all zeros (so next vz=0 always)
        # F[5,*] stays zero
        self.F = F

        # 4) No B (zero vertical accel is “baked in”)

        # 5) H (3×6) to measure [x, y, z]:
        H = np.zeros((3,6))
        H[0,0] = 1   # measure x
        H[1,1] = 1   # measure y
        H[2,2] = 1   # measure z
        self.H = H

        # 6) Q (6×6): process noise on v_x, v_y; no noise on v_z (vz=0 always)
        Q = np.zeros((6,6))
        Q[3,3] = process_var_vel_xy   # variance on vx
        Q[4,4] = process_var_vel_xy   # variance on vy
        # leave Q[5,5] = 0 so vz stays pinned at zero
        self.Q = Q

        # 7) R (3×3): measurement noise on (x,y,z)
        if isinstance(meas_var_pos, np.ndarray) and meas_var_pos.shape == (3, 3):
            self.R = meas_var_pos
        else:
            self.R = np.eye(3) * meas_var_pos

    def predict(self, u = None):
        """
        Predict step:
          x_{k+1|k} = F x_{k|k}
          P_{k+1|k} = F P F^T + Q
        Afterwards, we explicitly set vz=0 (in case of numerical drift).
        z_{k+1} remains whatever z_k was (until the next update).
        """
        # (1) Standard KF predict:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # (2) Pin vertical velocity to zero:
        self.x[5,0] = 0.0

        # (3) Zero out any covariance connections to vz:
        #     i.e. row 5 and column 5 → keep P[5,5] small (or zero)
        self.P[5, :] = 0
        self.P[:, 5] = 0
        # If you want to allow a tiny uncertainty in vz, you could instead do:
        #    self.P[5,5] = 1e-6
        # but setting it to zero means “we are sure vz=0.”

    def update(self, z):
        """
        Standard KF update with measurement z = [x_meas, y_meas, z_meas].
        Afterwards, re‐enforce vz = 0.
        """
        z = z.reshape((3,1))

        # 1) Innovation
        y = z - (self.H @ self.x)

        # 2) Innovation cov.
        S = self.H @ self.P @ self.H.T + self.R

        # 3) Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 4) State update
        self.x = self.x + (K @ y)

        # 5) Covariance update
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

        # 6) Pin vertical velocity to zero again:
        self.x[5,0] = 0.0
        # Zero out covariances involving vz:
        self.P[5, :] = 0
        self.P[:, 5] = 0


class IMM_KF: # This didn't quite work
    def __init__(self, model_filters, trans_matrix, mu_init):
        """
        IMM wrapper for M sub‐filters.

        Args:
          model_filters: list of M Kalman‐filter‐like objects, each having:
                         .x, .P, .F, .H, .Q, .R, .predict(u), .update(z)
          trans_matrix : M×M array where trans_matrix[j,i] = P(model_i at k | model_j at k-1)
          mu_init      : length‐M array of initial model probabilities (must sum to 1).
        """
        self.models = model_filters
        self.M = len(model_filters)
        self.Pij = np.array(trans_matrix)       # shape (M, M)
        self.mu = np.array(mu_init).astype(float)  # shape (M,)
        self.dim_x = self.models[0].dim_x
        self.dim_z = self.models[0].dim_z

    def mix_probabilities(self):
        mu_prev = self.mu.copy()  # shape (M,)
        c = np.zeros(self.M)
        for i in range(self.M):
            c[i] = np.sum(mu_prev * self.Pij[:, i])

        mix_w = np.zeros((self.M, self.M))  # mix_w[j,i]
        for i in range(self.M):
            for j in range(self.M):
                if c[i] > 0:
                    mix_w[j, i] = (mu_prev[j] * self.Pij[j, i]) / c[i]
                else:
                    mix_w[j, i] = 0

        mixed_states = []
        mixed_covs = []
        for i in range(self.M):
            x0_i = np.zeros((self.dim_x, 1))
            for j in range(self.M):
                x0_i += mix_w[j, i] * self.models[j].x

            P0_i = np.zeros((self.dim_x, self.dim_x))
            for j in range(self.M):
                dx = (self.models[j].x - x0_i)
                P0_i += mix_w[j, i] * (self.models[j].P + dx @ dx.T)

            mixed_states.append(x0_i)
            mixed_covs.append(P0_i)

        return mixed_states, mixed_covs

    def predict(self, controls):
        mixed_states, mixed_covs = self.mix_probabilities()
        for i in range(self.M):
            self.models[i].x = mixed_states[i]
            self.models[i].P = mixed_covs[i]
            # Each sub‐filter now does its own predict(u_i):
            self.models[i].predict(u=controls[i])

    def update(self, z):
        likelihoods = np.zeros(self.M)

        for i in range(self.M):
            x_pred = self.models[i].x.copy()
            P_pred = self.models[i].P.copy()

            z_vec = z.reshape((self.dim_z, 1))
            y = z_vec - (self.models[i].H @ x_pred)
            S = self.models[i].H @ P_pred @ self.models[i].H.T + self.models[i].R

            detS = np.linalg.det(S)
            if detS <= 0:
                likelihoods[i] = 0.0
            else:
                invS = np.linalg.inv(S)
                exponent = -0.5 * (y.T @ invS @ y).item()
                denom = np.sqrt((2*np.pi)**self.dim_z * detS)
                likelihoods[i] = np.exp(exponent) / denom

            K = P_pred @ self.models[i].H.T @ np.linalg.inv(S)
            self.models[i].x = x_pred + (K @ y)
            I = np.eye(self.dim_x)
            self.models[i].P = (I - K @ self.models[i].H) @ P_pred

        mu_pred = np.zeros(self.M)
        for i in range(self.M):
            mu_pred[i] = np.sum(self.mu * self.Pij[:, i])

        mu_post_unnorm = likelihoods * mu_pred
        total = np.sum(mu_post_unnorm)
        if total > 0:
            self.mu = mu_post_unnorm / total
        else:
            self.mu = mu_pred / np.sum(mu_pred)

    def fused_state(self):
        x_fuse = np.zeros((self.dim_x, 1))
        for i in range(self.M):
            x_fuse += self.mu[i] * self.models[i].x

        P_fuse = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.M):
            dx = self.models[i].x - x_fuse
            P_fuse += self.mu[i] * (self.models[i].P + dx @ dx.T)

        return x_fuse, P_fuse
    

# Helper functions

def project_to_world(preds, dataset, labels=None, camera = CAMERA):
    """
    Projects the predicted and ground truth ball positions from camera coordinates to world coordinates.
    
    Args:
        preds (torch.Tensor): Predicted ball positions in camera coordinates.
        labels (torch.Tensor): Ground truth ball positions in camera coordinates.
        dataset (Tracking3DVideoDataset): The dataset containing quantization information.
        
    Returns:
        tuple: Projected predicted and ground truth positions in world coordinates.
    """
    pred_pos = camera.project_ball_camera_to_world(preds*dataset.quantization, False)
    if labels is not None:
        truth_pos = camera.project_ball_camera_to_world(labels*dataset.quantization, False).numpy()
        return pred_pos, truth_pos
    else:
        return pred_pos, None

def project_to_camera(world_pos, dataset, camera = CAMERA):
    """
    Projects the world coordinates to camera coordinates.
    
    Args:
        world_pos (np.ndarray): Ball positions in world coordinates.
        dataset (Tracking3DVideoDataset): The dataset containing quantization information.
        
    Returns:
        np.ndarray: Projected ball positions in camera coordinates.
    """
    uvs, depths = camera.project_point(world_pos)
    radii = camera.get_radius_pixels(BALL_RADIUS, depths)

    q = dataset.label_quantization
    # ensure arrays
    u = (uvs[..., 0] // q).astype(int)
    v = (uvs[..., 1] // q).astype(int)
    r = (radii     // q).astype(int)

    # stack into (N,3) or (3,) if single
    out = np.stack([u, v, r], axis=-1)
    return out
    

def kalman_loop_out_of_fov(kf, pred_pos, truth_pos, last_measurement_ts = None):
    positions = []
    # Initialize Kalman filter with the first measurement
    # kf.predict()
    if last_measurement_ts is None: last_measurement_ts = len(pred_pos)
    kf.update(truth_pos[0])
    # Predict/update the next states
    for i, z in enumerate(pred_pos[1:], start=1):
        kf.predict()
        world_pos = kf.x.flatten()[:3]
        (x, y), depth = CAMERA.project_point(world_pos)
        if x < 0 or x > CAMERA.img_width or y < 0 or y > CAMERA.img_height:
            print(f"Ball went out of the FOV at timestep {i}, skipping update")
            print(f"Predicted position: {world_pos}, projected to image coordinates: ({x}, {y})")
            print(f"Real position: {truth_pos[i]}")
            print(f"Predicted position: {pred_pos[i]}")
            positions[i] = kf.x.flatten()[:3]
            continue
        if i < last_measurement_ts: kf.update(z)
        positions.append(kf.x.flatten()[:3])
        if kf.x.flatten()[:3][1] < 0: break # Stop when the ball is in the goal
    return kf, positions

def kalman_loop_in_fov(kf, pred_pos, truth_pos, last_measurement_ts = None, controls = None):
    positions = []
    # Initialize Kalman filter with the first measurement
    # kf.predict()
    if last_measurement_ts is None: last_measurement_ts = len(pred_pos)
    kf.update(truth_pos[0])
    # Predict/update the next states
    for i, z in enumerate(pred_pos[1:], start=1):
        if controls is not None:
            kf.predict(controls)
        else:
            kf.predict()
        if i < last_measurement_ts: kf.update(z)
        if controls is not None:
            position = kf.fused_state()[0][:3]
        else:
            position = kf.x.flatten()[:3]
        positions.append(position)
        if position[1] < 0: break # Stop when the ball is in the goal
    return kf, positions

def predict_until_goal(kf, positions, controls = None, truth = False):
    initial_n_positions = len(positions)
    if controls is not None:
        position = kf.fused_state()[0][:3]
    else:
        position = kf.x.flatten()[:3]
    while position[1] > 0: # The goal is at y=0
        if controls is not None:
            kf.predict(controls)
            position = kf.fused_state()[0][:3]
        else:
            kf.predict()
            position = kf.x.flatten()[:3]
        positions.append(position)
        if len(positions) > initial_n_positions + 1000:
            # print("Warning, many iterations. Maybe the ball is not going to the goal. ", "Ground truth" if truth else "Predicted")
            return kf, None
    return kf, positions

def predict_until_time(kf, positions, timesteps, controls = None):
    for ts in range(timesteps):
        if controls is not None:
            kf.predict(controls)
            position = kf.fused_state()[0][:3]
        else:
            kf.predict()
            position = kf.x.flatten()[:3]
        positions.append(position)
    return kf, positions


def kf_positions_for_idx(idx, dataset, all_preds, world_dataset, KFClass, cutoff, delta_t=0.01, process_var_acc=0.01, meas_var_pos=0.5): 
    tr = dataset.__gettr__(idx)
    preds = all_preds[idx]
    positions, truth_positions = kf_positions_for_tr(tr, dataset, preds, world_dataset, KFClass, cutoff, delta_t, process_var_acc, meas_var_pos)

    return np.array(positions), np.array(truth_positions)

def kf_positions_for_tr(tr, dataset, preds, world_dataset, KFClass, cutoff, delta_t=0.01, process_var_acc=0.01, meas_var_pos=0.5, print_graph=False): 
    # Get predictions
    _, labels, _ = dataset.__getitemtr__(tr)
    labels = labels.transpose(0, 1)

    if type(preds) is not np.ndarray:
        idx = dataset.__getidx__(tr)
        print(f"Using idx {idx} for trajectory {tr}")
        preds = preds[idx]

    # Get ground truth
    _, gt_labels, _ = world_dataset.__getitemtr__(tr)
    gt_labels = gt_labels.cpu().numpy()
    pred_pos, _ = project_to_world(preds, dataset)

    kf_preds = KFClass(delta_t, process_var_acc, meas_var_pos)
    kf_truth = KFClass(delta_t, 0.000001, 0.000001)

    last_measurement_ts = int(cutoff/delta_t)

    kf_preds, positions = kalman_loop_in_fov(kf_preds, pred_pos, gt_labels, last_measurement_ts)
    kf_truth, truth_positions = kalman_loop_in_fov(kf_truth, gt_labels, gt_labels)

    kf_preds, positions = predict_until_goal(kf_preds, positions, tr)
    kf_truth, truth_positions = predict_until_goal(kf_truth, truth_positions, tr)

    if print_graph:
        print_kf_graph(np.array(positions), np.array(truth_positions), pred_pos, gt_labels, last_measurement_ts)
        
    del kf_preds, kf_truth, preds
    return np.array(positions), np.array(truth_positions)

def print_kf_graph(positions, truth_positions, pred_pos, truth_labels, last_measurement_ts = None, ground_truth = False):
    if last_measurement_ts is None or last_measurement_ts > len(pred_pos): last_measurement_ts = len(pred_pos)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    last_pred_ts = len(positions)
    for i, label in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(positions[:, i], label='Kalman Filter', color='red')
        axs[i].plot(pred_pos[:last_measurement_ts, i], label='Predictions', color='blue')
        if ground_truth:
            if len(truth_labels) > last_pred_ts and positions[-1, 1] > 0.1:
                # axs[i].plot(truth_positions[:last_pred_ts, i], label='Ground Truth', color='green')
                axs[i].plot(truth_labels[:last_pred_ts, i], label='Ground Truth Labels', color='green')
            else:
                axs[i].plot(truth_positions[:, i], label='Ground Truth', color='green')
            # axs[i].plot(truth_labels[:last_pred_ts, i], label='Ground Truth Labels', color='orange')
        axs[i].axvline(x=last_measurement_ts, color='grey', linestyle='--', label='Last Measurement')
        axs[i].set_title(f'{label} Position')
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()


def majority_regime(pred_pos, last_measurement_ts = None, delta_t=0.01,
                    pv_xy=0.1, pv_z=0.1, mv=0.5):
    """
    Run the regime‐switching loop on one sequence of z‐positions `pred_pos`
    (shape N×3 or N×:z-index). Return the majority regime: "roll" or "throw".
    """
    roll_kf   = RollKF(delta_t, pv_xy, mv)
    throw_kf = ThrowKF(delta_t, pv_z,  mv)

    regime = 'roll'
    votes = []
    positions_throw = []
    positions_roll = []

    if last_measurement_ts is None or last_measurement_ts > len(pred_pos): last_measurement_ts = len(pred_pos)

    for i, meas in enumerate(pred_pos):
        # predict
        if i >= last_measurement_ts: break
        
        roll_kf.predict()
        throw_kf.predict()
        # if regime == 'flight':
        #     x_pred = flight_kf.x
        # else:
        #     x_pred = roll_kf.x


        # update

        z_meas  = meas[2]
        # vz_pred = x_pred[5,0]
        throw_kf.update(meas)
        roll_kf.update(meas)

        # hysteresis
        if regime == 'throw' and z_meas < T_DOWN:
            regime = 'roll'
        elif regime == 'roll' and z_meas > T_UP:
            regime = 'throw'

        votes.append(regime)
        positions_throw.append(throw_kf.x.flatten()[:3])
        positions_roll.append(roll_kf.x.flatten()[:3])
        
        # print(z_pred, regime)

    # majority vote
    rolls   = votes.count('roll')
    throws = votes.count('throw')
    # print(f"Votes: {rolls} rolls, {throws} throws")
    return (roll_kf, positions_roll, 'roll') if rolls > throws else (throw_kf, positions_throw, 'throw')


def evaluate_kf(dataloader, dataloader_world, model, device, KFClass = None, cutoff_time = None, cutoff_distance = None, time_pred = None, all_preds=None, check_result='goal'):
    """
    Evaluate Kalman Filter positions for all videos in the dataloader.
    Returns an array of differences between predicted and ground truth positions, and a confusion matrix.
    Trajectories that go "out" of the goal are skipped in the differences calculation.
    dataset = dataloader.dataset
    """

    if cutoff_time is None and cutoff_distance is None:
        print("No cutoff specified, looking at all positions.")
    elif cutoff_time is not None and cutoff_distance is not None:
        raise ValueError("Specify either cutoff_time or cutoff_distance, not both.")
    elif cutoff_time is not None:
        print(f"Using cutoff time: {cutoff_time} seconds.")
        cutoff_type = 'time'
    elif cutoff_distance is not None:
        print(f"Using cutoff distance: {cutoff_distance} meters.")
        cutoff_type = 'distance'
    
    if check_result == 'time':
        if time_pred is None:
            raise ValueError("If check_result is 'time', time_pred must be specified.")
        elif time_pred is not None:
            print(f"Using time prediction: {time_pred} seconds.")
    elif check_result == 'goal':
        if time_pred is not None:
            print(f"WARNING: time_pred is ignored when check_result is 'goal'.")
        else:
            print("Using goal prediction.")
    else:
        raise ValueError("check_result must be either 'goal' or 'time'.")
    

    dataset = dataloader.dataset
    N = len(dataset)
    diffs = []

    # Prepare confusion matrix
    classes = ["in", "out"]
    conf_mat = np.zeros((2,2), dtype=int)

    if all_preds is None: all_preds = network.get_preds_all_videos(model, dataloader, device=device, num_steps=model.training_params['num_steps'])
    for idx in tqdm(range(N), desc="Evaluating Kalman Filter"):
        _, labels, _ = dataset.__getitem__(idx)
        tr = dataset.__gettr__(idx)
        # Get predictions
        preds = all_preds[idx]
        labels = labels.transpose(0, 1)

        # Get ground truth
        _, gt_labels, _ = dataloader_world.dataset.__getitemtr__(tr)
        gt_labels = gt_labels.cpu().numpy()
        pred_pos, _ = project_to_world(preds, dataset)
        N = len(pred_pos)

        if cutoff_type == "time":
            last_measurement_ts = int(cutoff_time/DELTA_T)
            if last_measurement_ts > N:
                # print(f"Skipping trajectory {tr} due to insufficient measurements.")
                continue
        elif cutoff_type == "distance":
            # Calculate the last measurement based on the Y position
            y_positions = gt_labels[:, 1]
            excluded_preds = np.where(y_positions < cutoff_distance)
            # print(y_positions[0])
            # print(len(excluded_preds[0]), N, len(y_positions))
            if len(excluded_preds[0]) == len(y_positions):
                # print(f"Skipping trajectory {tr} due to no measurements within cutoff distance.")
                continue
            elif len(excluded_preds[0]) == 0:
                # print(f"Skipping trajectory {tr} due to not getting to cutoff distance.")
                # continue
                last_measurement_ts = N
            else:
                last_measurement_ts = excluded_preds[0][0]

        # print(f"Last measurement timestamp: {last_measurement_ts}, Total timesteps: {N}")
        
        if KFClass is None:
            kf_preds, kf_pred_positions, pred_regime = majority_regime(pred_pos, last_measurement_ts=last_measurement_ts)

            KFClass = RollKF if pred_regime == 'roll' else ThrowKF
        else:
            kf_preds = KFClass(DELTA_T, PROCESS_VAR_VEL_XY, MEAS_VAR_POS)
            kf_preds, kf_pred_positions = kalman_loop_in_fov(kf_preds, pred_pos, gt_labels, last_measurement_ts)
            
        kf_truth = KFClass(DELTA_T, PROCESS_VAR_VEL_XY, MEAS_VAR_POS)

        kf_truth, truth_positions = kalman_loop_in_fov(kf_truth, gt_labels, gt_labels, last_measurement_ts)

        # print(f"Predicted positions shape: {len(kf_pred_positions)}")
        # print(f"Ground truth positions shape: {len(truth_positions)}")

        if check_result == 'goal':
            kf_preds, kf_pred_positions = predict_until_goal(kf_preds, kf_pred_positions)
            kf_truth, truth_positions = predict_until_goal(kf_truth, truth_positions)
            if kf_pred_positions is None or truth_positions is None:
                # print(f"Skipping trajectory {tr} due to insufficient measurements for goal prediction.")
                continue
        elif check_result == 'time':
            timesteps_pred = int(time_pred / DELTA_T)
            # print(f"Predicting until {timesteps_pred} timesteps after the last measurement.")
            # print(f"Last measurement timestamp: {last_measurement_ts}, Total timesteps: {last_measurement_ts + timesteps_pred}")
            if last_measurement_ts + timesteps_pred > len(gt_labels):
                # print(f"Skipping trajectory {tr} due to insufficient measurements for time prediction.")
                continue
            kf_preds, kf_pred_positions = predict_until_time(kf_preds, kf_pred_positions, timesteps_pred)
            kf_truth, truth_positions = predict_until_time(kf_truth, truth_positions, timesteps_pred)
            
        kf_pred_positions = np.array(kf_pred_positions)
        # print(f"Predicted positions shape: {kf_pred_positions.shape}")
        truth_positions = np.array(truth_positions)
        # print(f"Ground truth positions shape: {truth_positions.shape}")
        
        final_prediction = kf_pred_positions[-1]
        final_true_position = truth_positions[-1]


        if check_result == 'goal':
            pred_type, gt_type = get_traj_type(final_prediction, dataset, tr, FIELD)

            # 1) Record confusion for *all* trajectories
            i = classes.index(gt_type)
            j = classes.index(pred_type)
            conf_mat[i, j] += 1

            # if pred_type == 'in' and gt_type == 'in':

            # Here record the results in a 3x3 confusion matrix ("in", "almost_in", "out")

            if pred_type == 'out' or gt_type == 'out':
                continue

        # print(f"Position in goal: {position_in_goal}, Truth in goal: {truth_in_goal}")
        
        # Add the timestamp to the final positions
        final_prediction = np.append(final_prediction, len(kf_pred_positions)*DELTA_T)
        final_true_position = np.append(final_true_position, len(truth_positions)*DELTA_T)

        diffs.append([abs(final_prediction[k] - final_true_position[k]) for k in range(len(final_prediction))])
        # diffs.append([final_prediction[k] - final_true_position[k] for k in range(len(final_prediction))])
        # 7) Delete everything large before next iteration
        # del preds, kf_pred_positions, truth_positions
        del final_prediction, final_true_position, pred_pos, gt_labels
        del preds, labels, kf_pred_positions, truth_positions, kf_preds, kf_truth
        torch.cuda.empty_cache()
        # break

    diffs = np.array(diffs, dtype=np.float32)
    # print_kf_graph(kf_pred_positions, truth_positions, pred_pos, gt_labels, last_measurement_ts)
    if check_result == 'goal':
        conf_df = pd.DataFrame(
            conf_mat,
            index=[f"True_{c}" for c in classes],
            columns=[f"Pred_{c}" for c in classes]
        )

        print("\nConfusion matrix:")
        print(conf_df)
        acc = np.trace(conf_mat) / conf_mat.sum()
        print(f"\nOverall accuracy: {acc:.3f}")
        return diffs, conf_mat, acc
    else:
        return diffs, None, None
    
def print_kf(dataloader, dataloader_world, model, device, KFClass = None, cutoff_time = None, cutoff_distance = None, time_pred = None, all_preds=None, check_result='goal', identifier = '', ground_truth = False, gen_video=False):
    if cutoff_time is None and cutoff_distance is None:
        print("No cutoff specified, looking at all positions.")
    elif cutoff_time is not None and cutoff_distance is not None:
        raise ValueError("Specify either cutoff_time or cutoff_distance, not both.")
    elif cutoff_time is not None:
        print(f"Using cutoff time: {cutoff_time} seconds.")
        cutoff_type = 'time'
    elif cutoff_distance is not None:
        print(f"Using cutoff distance: {cutoff_distance} meters.")
        cutoff_type = 'distance'
    
    if check_result == 'time':
        if time_pred is None:
            raise ValueError("If check_result is 'time', time_pred must be specified.")
        elif time_pred is not None:
            print(f"Using time prediction: {time_pred} seconds.")
    elif check_result == 'goal':
        if time_pred is not None:
            print(f"WARNING: time_pred is ignored when check_result is 'goal'.")
        else:
            print("Using goal prediction.")
    else:
        raise ValueError("check_result must be either 'goal' or 'time'.")
    

    dataset = dataloader.dataset
    N = len(dataset)
    diffs = []

    # Prepare confusion matrix
    classes = ["in", "out"]
    conf_mat = np.zeros((2,2), dtype=int)

    if all_preds is None: all_preds = network.get_preds_all_videos(model, dataloader, device=device, num_steps=10)
    for idx in range(N):
        video, labels, length = dataset.__getitem__(idx)
        tr = dataset.__gettr__(idx)
        # Get predictions
        preds = all_preds[idx]
        labels = labels.transpose(0, 1)

        # Get ground truth
        _, gt_labels, _ = dataloader_world.dataset.__getitemtr__(tr)
        gt_labels = gt_labels.cpu().numpy()
        pred_pos, _ = project_to_world(preds, dataset)
        N = len(pred_pos)

        if cutoff_type == "time":
            last_measurement_ts = int(cutoff_time/DELTA_T)
            if last_measurement_ts > N:
                # print(f"Skipping trajectory {tr} due to insufficient measurements.")
                continue
        elif cutoff_type == "distance":
            # Calculate the last measurement based on the Y position
            y_positions = gt_labels[:, 1]
            excluded_preds = np.where(y_positions < cutoff_distance)
            # print(y_positions[0])
            # print(len(excluded_preds[0]), N, len(y_positions))
            if len(excluded_preds[0]) == N-1:
                # print(f"Skipping trajectory {tr} due to no measurements within cutoff distance.")
                continue
            elif len(excluded_preds[0]) == 0:
                # print(f"Skipping trajectory {tr} due to not getting to cutoff distance.")
                # continue
                last_measurement_ts = N
            else:
                last_measurement_ts = excluded_preds[0][0]

        # print(f"Last measurement timestamp: {last_measurement_ts}, Total timesteps: {N}")
        
        if KFClass is None:
            kf_preds, kf_pred_positions, pred_regime = majority_regime(pred_pos, last_measurement_ts=last_measurement_ts)

            KFClass = RollKF if pred_regime == 'roll' else ThrowKF
        else:
            kf_preds = KFClass(DELTA_T, PROCESS_VAR_VEL_XY, MEAS_VAR_POS)
            kf_preds, kf_pred_positions = kalman_loop_in_fov(kf_preds, pred_pos, gt_labels, last_measurement_ts)
            
        kf_truth = KFClass(DELTA_T, PROCESS_VAR_VEL_XY, MEAS_VAR_POS)

        kf_truth, truth_positions = kalman_loop_in_fov(kf_truth, gt_labels, gt_labels, last_measurement_ts)

        # print(f"Predicted positions shape: {len(kf_pred_positions)}")
        # print(f"Ground truth positions shape: {len(truth_positions)}")

        if check_result == 'goal':
            kf_preds, kf_pred_positions = predict_until_goal(kf_preds, kf_pred_positions)
            kf_truth, truth_positions = predict_until_goal(kf_truth, truth_positions)
            if kf_pred_positions is None or truth_positions is None:
                # print(f"Skipping trajectory {tr} due to insufficient measurements for goal prediction.")
                continue
        elif check_result == 'time':
            timesteps_pred = int(time_pred / DELTA_T)
            # print(f"Predicting until {timesteps_pred} timesteps after the last measurement.")
            # print(f"Last measurement timestamp: {last_measurement_ts}, Total timesteps: {last_measurement_ts + timesteps_pred}")
            if last_measurement_ts + timesteps_pred > len(gt_labels):
                # print(f"Skipping trajectory {tr} due to insufficient measurements for time prediction.")
                continue
            kf_preds, kf_pred_positions = predict_until_time(kf_preds, kf_pred_positions, timesteps_pred)
            kf_truth, truth_positions = predict_until_time(kf_truth, truth_positions, timesteps_pred)
            
        kf_pred_positions = np.array(kf_pred_positions)
        # print(f"Predicted positions shape: {kf_pred_positions.shape}")
        truth_positions = np.array(truth_positions)
        # print(f"Ground truth positions shape: {truth_positions.shape}")
        break
        
    print_kf_graph(kf_pred_positions, truth_positions, pred_pos, gt_labels, last_measurement_ts, ground_truth=ground_truth)
    print_kf_3D(kf_pred_positions, truth_positions, pred_pos, labels, last_measurement_ts, ground_truth=ground_truth)
    if gen_video:
        video_path = gen_kf_video(
            video, kf_pred_positions, truth_positions, preds, labels,
            last_measurement_ts, show_ground_truth=ground_truth, identifier=identifier
        )
        print(f"Video saved to {video_path}")

def iter_kf_results(
    dataloader, dataloader_world, model, device,
    KFClass=None, cutoff_time=None, cutoff_distance=None,
    time_pred=None, all_preds=None, check_result='goal',
    ground_truth=False
):
    """
    Generator that yields one result per valid trajectory.
    Each yield returns a dict with everything needed to plot or export video.
    """

    if cutoff_time is None and cutoff_distance is None:
        print("No cutoff specified, looking at all positions.")
        cutoff_type = None
    elif cutoff_time is not None and cutoff_distance is not None:
        raise ValueError("Specify either cutoff_time or cutoff_distance, not both.")
    elif cutoff_time is not None:
        print(f"Using cutoff time: {cutoff_time} seconds.")
        cutoff_type = 'time'
    else:
        print(f"Using cutoff distance: {cutoff_distance} meters.")
        cutoff_type = 'distance'
    
    if check_result == 'time':
        if time_pred is None:
            raise ValueError("If check_result is 'time', time_pred must be specified.")
        else:
            print(f"Using time prediction: {time_pred} seconds.")
    elif check_result == 'goal':
        if time_pred is not None:
            print("WARNING: time_pred is ignored when check_result is 'goal'.")
        else:
            print("Using goal prediction.")
    else:
        raise ValueError("check_result must be either 'goal' or 'time'.")

    dataset = dataloader.dataset
    n_items = len(dataset)

    # precompute model preds once if not provided
    if all_preds is None:
        all_preds = network.get_preds_all_videos(model, dataloader, device=device, num_steps=10)

    for idx in range(n_items):
        # ---- fetch data ----
        video, labels, length = dataset.__getitem__(idx)
        tr = dataset.__gettr__(idx)

        preds = all_preds[idx]
        labels = labels.transpose(0, 1)

        # world GT for same trajectory
        _, gt_labels, _ = dataloader_world.dataset.__getitemtr__(tr)
        gt_labels = gt_labels.cpu().numpy()

        pred_pos, _ = project_to_world(preds, dataset)  # measurements in world
        N = len(pred_pos)

        # ---- determine last_measurement_ts based on cutoff ----
        if cutoff_type == "time":
            last_measurement_ts = int(cutoff_time / DELTA_T)
            if last_measurement_ts > N:
                continue  # insufficient measurements
        elif cutoff_type == "distance":
            y_positions = gt_labels[:, 1]
            excluded_preds = np.where(y_positions < cutoff_distance)
            if len(excluded_preds[0]) == N - 1:
                continue  # nothing within cutoff
            elif len(excluded_preds[0]) == 0:
                last_measurement_ts = N
            else:
                last_measurement_ts = excluded_preds[0][0]
        else:
            # no cutoff: use all available measurements
            last_measurement_ts = N

        # ---- Kalman filters for predicted measurements ----
        if KFClass is None:
            kf_preds, kf_pred_positions, pred_regime = majority_regime(pred_pos, last_measurement_ts=last_measurement_ts)
            KFClass_local = RollKF if pred_regime == 'roll' else ThrowKF
        else:
            KFClass_local = KFClass
            kf_preds = KFClass_local(DELTA_T, PROCESS_VAR_VEL_XY, MEAS_VAR_POS)
            kf_preds, kf_pred_positions = kalman_loop_in_fov(kf_preds, pred_pos, gt_labels, last_measurement_ts)

        # ---- Kalman filter driven by GT (reference trajectory) ----
        kf_truth = KFClass_local(DELTA_T, PROCESS_VAR_VEL_XY, MEAS_VAR_POS)
        kf_truth, truth_positions = kalman_loop_in_fov(kf_truth, gt_labels, gt_labels, last_measurement_ts)

        # ---- extend to goal or to fixed time horizon ----
        if check_result == 'goal':
            kf_preds, kf_pred_positions = predict_until_goal(kf_preds, kf_pred_positions)
            kf_truth, truth_positions = predict_until_goal(kf_truth, truth_positions)
            if kf_pred_positions is None or truth_positions is None:
                continue  # not enough info to make a goal prediction
        else:  # check_result == 'time'
            timesteps_pred = int(time_pred / DELTA_T)
            if last_measurement_ts + timesteps_pred > len(gt_labels):
                continue
            kf_preds, kf_pred_positions = predict_until_time(kf_preds, kf_pred_positions, timesteps_pred)
            kf_truth, truth_positions = predict_until_time(kf_truth, truth_positions, timesteps_pred)

        # ---- package results ----
        result = {
            "idx": idx,
            "tr": tr,
            "video": video,
            "labels": labels,
            "preds_logits_or_raw": preds,  # keep raw preds if you need to reproject/inspect
            "pred_pos": np.array(pred_pos),
            "gt_labels": gt_labels,
            "kf_pred_positions": np.array(kf_pred_positions),
            "truth_positions": np.array(truth_positions),
            "last_measurement_ts": last_measurement_ts,
            "ground_truth": ground_truth,
        }

        yield result


# Convenience: plot a single result dict (non-blocking optional)
def show_kf_result(result, identifier='', gen_video=False):
    print_kf_graph(
        result["kf_pred_positions"], result["truth_positions"],
        result["pred_pos"], result["gt_labels"],
        result["last_measurement_ts"], ground_truth=result["ground_truth"]
    )
    print_kf_3D(
        result["kf_pred_positions"], result["truth_positions"],
        result["pred_pos"], result["labels"],
        result["last_measurement_ts"], ground_truth=result["ground_truth"]
    )
    if gen_video:
        path = gen_kf_video(
            result["video"], result["kf_pred_positions"],
            result["truth_positions"], result["preds_logits_or_raw"],
            result["labels"], result["last_measurement_ts"],
            show_ground_truth=result["ground_truth"], identifier=identifier
        )
        print(f"Video saved to {path}")

def print_kf_3D(kf_pred_positions, truth_positions, pred_pos, gt_labels, last_measurement_ts=None, ground_truth=False):
    fig, ax = FIELD.plot()
    ax.plot(pred_pos[:last_measurement_ts, 0], pred_pos[:last_measurement_ts, 1], pred_pos[:last_measurement_ts, 2], color='blue', label='Measurements')
    last_pred_ts = len(kf_pred_positions)
    ax.plot(kf_pred_positions[:, 0], kf_pred_positions[:, 1], kf_pred_positions[:, 2], color='red', label='Kalman Filter Predictions')
    if ground_truth:
        if len(gt_labels) > last_pred_ts:
            ax.plot(gt_labels[:last_pred_ts, 0], gt_labels[:last_pred_ts, 1], gt_labels[:last_pred_ts, 2], color='orange', label='Ground Truth Labels')
        else:
            ax.plot(truth_positions[:, 0], truth_positions[:, 1], truth_positions[:, 2], color='green', label='Ground Truth')
    plt.show()

def gen_kf_video(
    video: torch.Tensor,
    kf_pred_positions: np.ndarray,
    truth_positions: np.ndarray,
    dataset: dt.Tracking3DVideoDataset,
    preds_cam: np.ndarray,
    labels_cam: np.ndarray,
    last_measurement_ts: int,
    identifier: str = '',
    show_ground_truth: bool = True,
    fps: int = 30,
) -> str:
    """
    Generate and save a video showing the Kalman Filter predictions and ground truth.

    Draws measured positions (red), KF predictions (green), and optionally
    true positions (yellow) on top of the original frames.

    Args:
        video (torch.Tensor): Tensor of shape (T, C, H, W), values in [0,1].
        kf_pred_positions (np.ndarray): KF predictions in world coords.
        truth_positions (np.ndarray): True positions in world coords.
        preds_cam (np.ndarray): Predicted positions in camera coords, shape (T,3).
        labels_cam (np.ndarray): Measured positions in camera coords, shape (T,3).
        last_measurement_ts (int): Number of frames to display.
        show_ground_truth (bool): If True, overlay true positions too.
        fps (int): Output video frame rate.

    Returns:
        str: Path to the saved video file.
    """
    if identifier != '':
        identifier = f"_{identifier}"
    # Project the KF world-coords into camera space
    kf_positions_cam = project_to_camera(kf_pred_positions, dataset)
    n = len(kf_positions_cam)
    # Choose which ground-truth to overlay, truncated to match KF length
    if len(labels_cam) >= n:
        gt_cam = labels_cam[:n]
    else:
        gt_cam = project_to_camera(truth_positions, dataset)[:n]

    out_size = (1280, 720)  # Fixed output size for the video

    # Determine frame size from the torch tensor
    # video: (T, C, H, W)
    T, C, H, W = video.shape
    scale = out_size[0] / W  # Scale factor to fit the video size
    if n > T:
        kf_positions_cam = kf_positions_cam[:T]
        gt_cam = gt_cam[:T]
    if last_measurement_ts > T:
        last_measurement_ts = T

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_path = f'kf_output{identifier}.avi'
    writer = cv2.VideoWriter(out_path, fourcc, fps, out_size)

    def S(x): return int(x * scale)

    for i in range(T):
        # Pull frame, convert to H×W×3 uint8 BGR
        frame = video[i]  # C×H×W
        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if C == 2:
            frame_np = np.concatenate([frame_np[:, :, 0:1], np.zeros((H, W, 1), dtype=np.uint8), frame_np[:, :, 1:2]], axis=-1)
        if C == 1:
            frame_np = cv2.cvtColor(frame_np[:, :, 0], cv2.COLOR_GRAY2BGR)

        frame_np = cv2.resize(frame_np, out_size, interpolation=cv2.INTER_NEAREST)

        frame_np = np.ascontiguousarray(frame_np)
        

        # Draw KF prediction in GREEN
        xk, yk, rk = kf_positions_cam[i]
        cv2.circle(frame_np, (S(xk), S(yk)), S(rk), (0, 255, 0), 2)

        # Draw (raw) filter’s own preds_cam in BLUE for comparison
        if i < last_measurement_ts:
            xp, yp, rp = preds_cam[:, i]
            cv2.circle(frame_np, (S(xp), out_size[1] - S(yp)), S(rp), (255, 0, 0), 2)
            cv2.putText(
                frame_np,
                text=f"Measuring",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=(0, 255, 0),   # Green text
                thickness=2,
                lineType=cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame_np,
                text=f"Predicting",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=(255, 0, 0),   # Blue text
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # print(i, "Predicting")
        # Optionally draw the true positions in YELLOW
        if show_ground_truth:
            xg, yg, rg = gt_cam[i]
            cv2.circle(frame_np, (S(xg), S(yg)), S(rg), (0, 255, 255), 2)
        writer.write(frame_np)

    writer.release()
    return out_path
    


    

def get_traj_type(position_in_goal, dataset, tr, field):
    row = dataset.trajectories[dataset.trajectories["tr"] == tr]
    if row.empty:
        raise ValueError(f"Trajectory {tr} not found in the dataset.")
    goes_in = row["goes_in"].iloc[0]
    gt_type = "in" if goes_in == 1 else "out"
    if position_in_goal is not None: 
        x_y0, _, z_y0 = position_in_goal
    else:
        return "out", gt_type
    params = {
        "x_y0": x_y0,
        "z_y0": z_y0,
        "field": field
    }
    constraints_goes_in = [
        lambda params: - (params["field"].GW/2) < params["x_y0"] - params["field"].center[0] < params["field"].GW/2,  # Ball should go through the goal or near it
        lambda params: 0 <= params["z_y0"] < params["field"].GH,  # Ball should go through the goal or near it
    ]
    margin = 0
    constraints_goes_out = [
        lambda params: - (params["field"].GW/2 + margin) > params["x_y0"] - params["field"].center[0] or
            params["x_y0"] - params["field"].center[0] > params["field"].GW/2 + margin or
            -margin > params["z_y0"] or
            params["z_y0"] > params["field"].GH + margin
    ]

    if all(constraint(params) for constraint in constraints_goes_in):
        pred_type = "in"
    elif all(constraint(params) for constraint in constraints_goes_out):
        pred_type = "out"
    else:
        raise ValueError(f"Trajectory type for position {position_in_goal} and row {row} is not recognized. Something is wrong")
    
    return pred_type, gt_type
