import numpy as np

class CA3DKalmanFilter:
    def __init__(self, dt, process_var_acc, meas_var_pos,
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
        self.R = np.eye(3) * meas_var_pos

    def predict(self):
        # x = F x
        self.x = self.F @ self.x
        # P = F P F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        z: (3,) or (3,1) measurement vector [x,y,z]
        """
        z = z.reshape((3,1))
        # Innovation
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # State update
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

# Example usage:
dt = 0.1
process_var_acc = 0.01
meas_var_pos = 0.5
initial_state = np.array([0,0,0, 0,0,0, 0,0,-9.81])

kf = CA3DKalmanFilter(dt, process_var_acc, meas_var_pos, initial_state)

# Simulated 3D parabolic trajectory (replace with your SNN outputs)
times = np.arange(0, 1, dt)
measurements = np.vstack([
    5 * times,         # x = 5 t
    3 * times,         # y = 3 t
    -4.9 * times**2    # z = -½ g t^2
]).T

for z in measurements:
    kf.predict()
    kf.update(z)

print("Estimated state (pos,vel,acc):")
print(kf.x.flatten())
