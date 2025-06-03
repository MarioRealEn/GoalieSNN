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
