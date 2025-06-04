import numpy as np

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
    6D KF in 3D with KNOWN a_z = -9.8 (free‐fall):
      state = [x, y, z, vx, vy, vz]^T.
      control u_k = a_z = -9.8 each step.
    Measurement: [x, y, z].
    """
    def __init__(self, dt, process_var_vel, meas_var_pos,
                 initial_state=None, initial_covariance=1e3):
        """
        dt               : time step (seconds or frames).
        process_var_vel  : variance on v_x, v_y, v_z.
        meas_var_pos     : variance on x,y,z measurements.
        initial_state    : optional 6×1 array [x,y,z,vx,vy,vz].
        initial_covariance: scalar to initialize P = I * initial_covariance.
        """
        self.dt = dt
        self.dim_x = 6
        self.dim_z = 3

        # 1) State vector x (6×1)
        self.x = np.zeros((6,1))
        if initial_state is not None:
            self.x = initial_state.reshape((6,1))

        # 2) Covariance P (6×6)
        self.P = np.eye(self.dim_x) * initial_covariance

        # 3) F (6×6): same constant‐velocity block (no accel in state)
        F = np.zeros((6,6))
        for i in range(3):
            F[i,   i]     = 1
            F[i,   3+i]   = dt
            F[3+i, 3+i]   = 1
        self.F = F

        # 4) B (6×1) maps the KNOWN a_z into Δz, Δv_z
        self.B = np.array([
            [0.0],             # no direct Δx from a_z
            [0.0],             # no direct Δy from a_z
            [0.5 * dt**2],     # Δz = ½·a_z·dt²
            [0.0],             # no Δvx from a_z
            [0.0],             # no Δv_y from a_z
            [dt]               # Δv_z = a_z·dt
        ])

        # 5) H (3×6): measure x,y,z
        H = np.zeros((3,6))
        H[0,0] = 1
        H[1,1] = 1
        H[2,2] = 1
        self.H = H

        # 6) Q (6×6): process noise on velocities
        Q = np.zeros((6,6))
        Q[3,3] = process_var_vel
        Q[4,4] = process_var_vel
        Q[5,5] = process_var_vel
        self.Q = Q

        # 7) R (3×3): measurement noise on x,y,z
        self.R = np.eye(3) * meas_var_pos

        # 8) We’ll always pass u = -9.8 for free‐fall
        self.u_const = -9.8

    def predict(self, u=None):
        """
        Predict step with known vertical acceleration u_k.
        If u is None, we default to -9.8.
        """
        if u is None:
            u = self.u_const

        # State‐predict
        self.x = (self.F @ self.x) + (self.B * u)

        # Covariance‐predict
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
        self.R = np.eye(3) * meas_var_pos

    def predict(self):
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