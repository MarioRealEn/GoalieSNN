import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
import cv2
import os
import pandas as pd
import torch


BALL_RADIUS = 0.11

class TrajectoryGenerator():
    def __init__(self, constraints, ball_radius, field, Vmax, camera = None):
        self.constraints = constraints
        self.Vmax = Vmax
        self.Vmin = 2
        self.g = 9.81
        self.Crr = 0.01 # Coefficient of rolling resistance
        self.field = field
        self.R = ball_radius
        # self.t_max_roll = 10
        self.t_min_roll = 1
        self.camera = camera
        self.g_x_grid = None
        self.g_y_grid = None
 
    def get_velocities_throw(self, r0, plot = True, n = None):

        # Parameters
        c = 0.25       # scaling factor for velocity components

        # Number of sample points for velocity (spherical coordinates)
        num_v, num_theta, num_phi = 30, 30, 60

        # Create arrays for the speed magnitude v, polar angle theta and azimuth phi.
        v_array = np.linspace(self.Vmin, self.Vmax, num_v)
        # theta is chosen so that v0_z > 0; we avoid the extremes.
        theta_array = np.linspace(0.001, np.pi/2 - 0.001, num_theta)
        phi_array = np.linspace(0, 2*np.pi, num_phi)

        # Create a random order of the velocity vectors
        v_sphere = np.array(list(itertools.product(v_array, theta_array, phi_array)))
        np.random.shuffle(v_sphere)

        # We'll store the landing positions (in positive coordinates) and vertical speed.
        X, Y, Z = [], [], []  # For visualization
        valid_velocities, goal_hits, t_flights, t_goals = [], [], [], []
        cont = 0

        for v, theta, phi in v_sphere:
            # Convert spherical coordinates to Cartesian components:
            v0_x = v * np.sin(theta) * np.cos(phi)
            v0_y = v * np.sin(theta) * np.sin(phi)
            v0_z = v * np.cos(theta)
            if v0_z <= 0:  # ensure positive vertical component
                continue
            # Compute landing displacement (relative to r0)
            t_flight = 2 * v0_z / self.g
            dx = v0_x * t_flight
            dy = v0_y * t_flight

            # Landing constraint: the landing point must be within [0, L] and [0, W]
            # The landing point (r0 + displacement) is:
            landing_x = r0[0] + dx
            landing_y = r0[1] + dy

            # Compute time to cross y = 0 plane
            t_y0 = -r0[1] / v0_y if abs(v0_y) > 1e-5 else np.inf

            # Compute x and z
            x_y0 = r0[0] + t_y0 * v0_x
            z_y0 = r0[2] + t_y0 * v0_z - 0.5 * self.g * t_y0**2

            
            
            # Create params dict: params: r0, v0, lx, ly, tf, field, R, t_y0, y_y0, z_y0
            params = {
                'r0': r0,
                'v0': (v0_x, v0_y, v0_z), 
                'lx': landing_x,
                'ly': landing_y,
                'tf': t_flight,
                'field': self.field,
                'R': self.R,
                't_y0': t_y0,
                'x_y0': x_y0,
                'z_y0': z_y0
            }

            if all(constraint(params)
                   for constraint in self.constraints):
                # Instead of storing scaled velocity components, we store the landing
                # position in the positive domain along with the vertical velocity.
                X.append(r0[0] + c * v0_x)
                Y.append(r0[1] + c * v0_y)
                Z.append(c * v0_z)
                valid_velocities.append([v0_x, v0_y, v0_z])
                goal_hits.append([x_y0, 0, z_y0])
                t_flights.append(t_flight)
                t_goals.append(t_y0)
                cont += 1
                if (n is not None) and (cont >= n):
                    break

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        valid_velocities = np.array(valid_velocities)
        goal_hits = np.array(goal_hits)
        t_goals = np.array(t_goals)
        # Set up a 3D plot
        if plot:
            
            fig, ax = self.field.plot()

            if n is None:
            # Use triangulation to plot a surface from the velocity vectors
                tri = mtri.Triangulation(X, Y)
                surf = ax.plot_trisurf(tri, Z, cmap='viridis', edgecolor='none', zsort='min', zorder = 1)
                fig.colorbar(surf, shrink=0.5, aspect=5)
            else:
                # Plot the landing spots on the field
                ax.scatter(goal_hits[:,0], goal_hits[:,1], 0, c='red', s=self.R, label='Landing Spots', alpha=0.5, zorder=-1)

            # Labeling the axes; here the horizontal axes are in meters (landing position)
            # and the vertical axis shows the corresponding v0_z (m/s).
            ax.set_title('Landing Field and Allowed Velocities')

            plt.show()

        return valid_velocities, goal_hits, t_flights, t_goals
    
    def get_velocities_roll(self, r0, plot = True, n = None):

        # Parameters
        c = 0.25       # scaling factor for velocity components
        # threshold = 1e-2  # Threshold for stopping time

        # Number of sample points for velocity (spherical coordinates)
        num_v, num_phi = 30, 60

        # Create arrays for the speed magnitude v, and azimuth phi.
        v_array = np.linspace(self.Vmin, self.Vmax, num_v)
        phi_array = np.linspace(0, 2*np.pi, num_phi)

        # Create a random order of the velocity vectors
        v_polar = np.array(list(itertools.product(v_array, phi_array)))
        np.random.shuffle(v_polar)

        # We'll store the landing positions (in positive coordinates) and vertical speed.
        X, Y, Z = [], [], []  # For visualization
        valid_velocities, goal_hits, t_flights, t_goals = [], [], [], []
        cont = 0

        for v, phi in v_polar:
            # Convert spherical coordinates to Cartesian components:
            v0_x = v * np.cos(phi)
            v0_y = v * np.sin(phi)
            v0 = np.array([v0_x, v0_y, 0])
            # Compute landing displacement (relative to r0)
            t_stop = self.t_of_v(v, 0, -self.Crr * self.g)
            # if t_stop < self.t_min_roll or t_stop > self.t_max_roll: I think we don't need this, with just the min and max v is enough
            #     continue
            # print('t_stop:', t_stop)
            t_f = t_stop
            landing_x, landing_y, _ = self.s_of_t_roll(r0, v0, t_stop)
            

            # Compute time to cross y = 0 plane
            t_y0 = self.t_of_s_roll(r0[1], v0_y, v0, 0)

            if t_y0 is None or t_y0 < 0:
                continue

            # Compute x
            x_y0, y_y0, z_y0 = self.s_of_t_roll(r0, v0, t_y0)
            # print('y_y0:', y_y0)
            
            
            # Create params dict: params: r0, v0, lx, ly, tf, field, R, t_y0, y_y0, z_y0
            # print(type(r0), type(v0_x), type(v0_y), type(landing_x), type(landing_y), type(t_f), type(self.field), type(self.R), type(t_y0), type(x_y0))
            params = {
                'r0': r0,
                'v0': (v0_x, v0_y), 
                'lx': landing_x,
                'ly': landing_y,
                'tf': t_f,
                'field': self.field,
                'R': self.R,
                't_y0': t_y0,
                'x_y0': x_y0,
                'z_y0': self.R
            }


            if all(constraint(params)
                   for constraint in self.constraints):
                X.append(r0[0] + c * v0_x)
                Y.append(r0[1] + c * v0_y)
                Z.append(0)
                valid_velocities.append([v0_x, v0_y, 0])
                # landing_spots.append([landing_x, landing_y, self.R])
                goal_hits.append([x_y0, 0, self.R])
                t_flights.append(t_f)
                t_goals.append(t_y0)
                cont += 1
                if (n is not None) and (cont >= n):
                    break

        if n == None:
            X = np.array(X)
            Y = np.array(Y)
            Z = np.array(Z)
        valid_velocities = np.array(valid_velocities)
        goal_hits = np.array(goal_hits)
        t_goals = np.array(t_goals)
        # Set up a 3D plot
        if plot:
            
            fig, ax = self.field.plot()

            if n is None:
            # Use triangulation to plot a surface from the velocity vectors
                tri = mtri.Triangulation(X, Y)
                surf = ax.plot_trisurf(tri, Z, cmap='viridis', edgecolor='none', zsort='min', zorder = 1)
                fig.colorbar(surf, shrink=0.5, aspect=5)
            else:
                # Plot the landing spots on the field
                ax.scatter(goal_hits[:,0], goal_hits[:,1], 0, c='red', s=self.R, label='Landing Spots', alpha=0.5, zorder=-1)

            # Labeling the axes; here the horizontal axes are in meters (landing position)
            # and the vertical axis shows the corresponding v0_z (m/s).
            ax.set_title('Landing Field and Allowed Velocities')

            plt.show()

        return valid_velocities, goal_hits, t_flights, t_goals
    
    def s_of_t_roll(self, s0, v0, t):
        # return s0 + v0/self.mu * (1 - np.exp(-self.mu * t))
        # print(type(s0), type(v0), type(t))
        s = s0 + v0*t - 0.5 * self.Crr * self.g * t**2 * v0 / np.linalg.norm(v0)
        return s
    def t_of_s_roll(self, s0_i, v0_i, v0, s):
        if type(s0_i) is np.ndarray:
            raise TypeError("s0_i should be a scalar")
        a = 0.5 * self.Crr * self.g * v0_i / np.linalg.norm(v0)
        b = -v0_i
        c = s - s0_i

        # Find roots using np.roots
        roots = np.roots([a, b, c])
        # print(roots)
        
        # Filter for real roots that are >= 0
        valid_roots = [root.real for root in roots if np.isreal(root) and root.real >= 0]
        
        if not valid_roots:
            return None
        
        valid_root = min(valid_roots)
        return valid_root
    def s_of_t_throw(self, s0, v0, t):
        return s0 + v0 * t - 0.5 * self.g * t**2*np.array([0, 0, 1])
    def t_of_s_throw(self, s0, v0, s, ascending = True):
        """
        Given initial position s0 and initial velocity v0 (both numpy arrays)
        and a desired final position s (a numpy array),
        solve for the time t required for the projectile (thrown ball) to reach s,
        using the vertical (z) component of the trajectory:
        
        s_z = s0_z + v0_z*t - 0.5 * g * t^2
        
        Returns the smallest positive t if a real solution exists; otherwise, returns None.
        """
        s0_z = s0[2]
        s_z = s[2]
        v0_z = v0[2]

        # Coefficients for the quadratic equation: a*t^2 + b*t + c = 0
        a = 0.5 * self.g
        b = -v0_z
        c = s_z - s0_z

        # Find roots using np.roots
        roots = np.roots([a, b, c])
        
        # Filter for real roots that are >= 0
        valid_roots = [root.real for root in roots if np.isreal(root) and root.real >= 0]
        
        if not valid_roots:
            return None
        
        valid_root = min(valid_roots) if ascending else max(valid_roots)
        return valid_root
    def v_of_t(self, v0, t, a):
        # v = v0 + a*t
        return v0 + a*t
    def t_of_v(self, v0, v, a):
        # t = (v - v0)/a
        return (v - v0)/a
    def get_init_conditions(self, n, roll = False):
        # Get Rs
        x = np.linspace(0, self.field.W, 100)
        y = np.linspace(0, self.field.L, 100)
        z = [self.R]
        r_combs = np.array(list(itertools.product(x, y, z)))
        rs, vs, ls, ts, tgs = [], [], [], [], []
        while True:
            r = np.random.choice(r_combs.shape[0], 1, replace=False)
            r = r_combs[r][0]
            if (- self.field.GAW/2 < r[0] - self.field.center[0] < self.field.GAW/2 and
                    r[1] < self.field.GAL): # Make sure the ball is not in the goal area
                continue
            if self.camera is not None:
                if not self.camera.is_point_in_field_of_view(r):
                    continue
            v, l, t_flight, t_goal = self.get_velocities_throw(r, plot = False, n = 1) if not roll else self.get_velocities_roll(r, plot = False, n = 1)
            if len(v) == 0:
                continue
            rs.append(r)
            vs.append(v[0])
            ls.append(l[0])
            ts.append(t_flight)
            tgs.append(t_goal)
            if len(vs) == n:
                rs = np.array(rs)
                vs = np.array(vs)
                ls = np.array(ls)
                ts = np.array(ts)
                tgs = np.array(tgs)
                break
        return rs, vs, ls, ts, tgs
    
    def sample_trajectory(self, r0, v0, t_max, fps=None, roll = False):

        if fps is None and self.camera is not None:
            fps = self.camera.fps
        elif fps is None and self.camera is None:
            raise ValueError("FPS must be provided if no camera is defined.")
        
        num_frames = int(t_max // (1/fps))  # Compute the number of frames as an integer
        t_values = np.linspace(0, t_max, num_frames, endpoint=False)

        # Compute world positions along the trajectory using standard kinematics:
        trajectory = []
        for t in t_values:
            x, y, z = self.s_of_t_throw(r0, v0, t) if not roll else self.s_of_t_roll(r0, v0, t)
            trajectory.append([x, y, z])
        trajectory = np.array(trajectory).squeeze()
        return trajectory
    
    def plot_trajectory(self, trajectory, ax, camera=None):
        ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label='Trajectory')
        
        def draw_cross(ax, x, y, z=0, size=self.R, color='red', lw=0.5):
            """
            Draws a 2D cross (aligned to the ground) at a given point (x, y, z) in a 3D plot.
            
            Parameters:
                ax   - Matplotlib 3D axis
                x, y, z - Coordinates of the center of the cross
                size - Half-length of the cross lines
                color - Color of the cross
                lw    - Line width
            """
            ax.plot([x - size, x + size], [y, y], [z, z], color=color, linewidth=lw)  # Horizontal line
            ax.plot([x, x], [y - size, y + size], [z, z], color=color, linewidth=lw)  # Vertical line

        # Projected landing points on the floor
        start_proj = trajectory[0, 0], trajectory[0, 1], 0  # (x0, y0, ground)
        end_proj = trajectory[-1, 0], trajectory[-1, 1], 0  # (x_final, y_final, ground)

        # Draw crosses on the floor at the start and end projection points
        draw_cross(ax, *start_proj, color='black')  # Black cross for start
        draw_cross(ax, *end_proj, color='red')      # Red cross for landing point

        if camera is not None:
            camera.plot(ax)

    def generate_df_init_conditions(self, n, throw_ratio = 0.5, n_goal_bins_x = 5, n_goal_bins_y = 2): # This would generate the velocities for rolling and throwing and mix them

        self.g_x_grid = np.linspace(0, self.field.GW, n_goal_bins_x + 1)
        self.g_y_grid = np.linspace(0, self.field.GH, n_goal_bins_y + 1)
        unique_conditions = set()
        cont = 0
        all_init_conditions = pd.DataFrame(columns = ['tr', 'r0_x', 'r0_y', 'r0_z', 'v0_x', 'v0_y', 'v0_z', 'g_x', 'g_y', 'g_x_bin', 'g_y_bin', 'tf', 'is_roll'])
        while len(unique_conditions) < n:
            if np.random.rand() > throw_ratio:
                is_roll = True
            else:
                is_roll = False
            init_conditions = self.get_init_conditions(1, roll = is_roll)
            init_conditions = [np.squeeze(var) for var in init_conditions]
            init_conditions.append(is_roll)
            init_parts = []
            for item in init_conditions:
                # Convert item to a NumPy array (in case it isn't one), flatten it, round, and then convert to tuple.
                arr = np.array(item).flatten()
                init_parts.append(tuple(arr.tolist()))
            init_conditions_tuple = tuple(init_parts)
            if init_conditions_tuple not in unique_conditions:
                unique_conditions.add(init_conditions_tuple)
                # Goal bins implementation
                g_x = init_conditions[2][0]
                g_y = init_conditions[2][2]
                g_x_from_goal = g_x - self.field.W/2 + self.field.GW/2
                g_x_bin = np.digitize(g_x_from_goal, self.g_x_grid)
                g_y_bin = np.digitize(g_y, self.g_y_grid)

                new_row = {
                    'tr': int(len(all_init_conditions)),
                    'r0_x': init_conditions[0][0],
                    'r0_y': init_conditions[0][1],
                    'r0_z': init_conditions[0][2],
                    'v0_x': init_conditions[1][0],
                    'v0_y': init_conditions[1][1],
                    'v0_z': init_conditions[1][2],
                    'g_x': g_x,
                    'g_y': g_y,
                    'g_x_bin': g_x_bin,
                    'g_y_bin': g_y_bin,
                    'tf': init_conditions[3],
                    'tg': init_conditions[4],
                    'is_roll': is_roll
                }
                all_init_conditions = pd.concat([all_init_conditions, pd.DataFrame(new_row, index = [0])], ignore_index = True)

            cont += 1
            if cont > 10000:
                print('Warning: Tried to generate 10000 init conditions but only got', len(all_init_conditions))

        return all_init_conditions
    
    def generate_df_positions(self, init_conditions):
        if self.camera is not None:
            fps = self.camera.fps
        else:
            raise ValueError("A camera must be defined, as the df will contain the relative radius of the ball.")
        
        df = pd.DataFrame(columns = ['tr', 'frame', 'x', 'y', 'z', 'R_cam'])
        for _, row in init_conditions.iterrows():
            tr = row['tr']
            r0 = np.array([row['r0_x'], row['r0_y'], row['r0_z']])
            v0 = np.array([row['v0_x'], row['v0_y'], row['v0_z']])
            t_flight = row['tf']
            is_roll = row['is_roll']
            trajectory = self.sample_trajectory(r0, v0, t_flight, fps, roll = is_roll)
            
            # Add the trajectory to the DataFrame
            for j, pos in enumerate(trajectory):
                # Define vector perpendicular to the position of the ball and of the camera
                # and project it to the ball plane
                # This is the vector from the camera to the ball
                # theta, phi = self.camera.orientation
                # # Forward vector from the camera
                # forward = np.array([
                #     np.cos(theta) * np.cos(phi),
                #     np.cos(theta) * np.sin(phi),
                #     np.sin(theta)
                # ])
                # # Unitary vector perpendicular to forward
                # v_perp = np.array([np.tan(-phi)/(1+np.tan(-phi)), 1/(1+np.tan(-phi)), 0])
                # v_perp = v_perp / np.linalg.norm(v_perp)  # Normalize the vector
                # print('This should be zero:', forward.dot(v_perp))
                # print('This should be one:', np.linalg.norm(v_perp))
                pos_cam, depth = self.camera.project_point(pos)
                if pos_cam is None:
                    r_cam1 = np.nan
                    x_cam = np.nan
                    y_cam = np.nan
                    # r_cam2 = np.nan
                else:
                    r_cam1 = self.camera.get_radius_pixels(self.R, depth)
                    x_cam = pos_cam[0]
                    y_cam = pos_cam[1]
                    # r_cam2_vec = pos_cam - self.camera.project_point(pos + v_perp * self.R)[0]
                    # r_cam2 = np.linalg.norm(r_cam2_vec)
                new_row = {
                    'tr': tr,
                    'frame': j,
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2],
                    'x_cam': x_cam,
                    'y_cam': y_cam,
                    'R_cam': r_cam1,
                    # 'R_cam2': r_cam2
                }
                df = pd.concat([df, pd.DataFrame(new_row, index = [0])], ignore_index = True)
        return df

    
    # Data Frame distribution plotting functions
    def plot_r_distribution(self, df):
        fig, ax = self.field.plot()
        for _, row in df.iterrows():
            r0 = row[['r0_x', 'r0_y', 'r0_z']].values
            ax.scatter(r0[0], r0[1], c='r', s=20, marker='x')

    def plot_v_distribution(self, df):
        v0_abs = []
        for _, row in df.iterrows():
            v0 = row[['v0_x', 'v0_y', 'v0_z']].values
            v0_abs.append(np.linalg.norm(v0))
        plt.figure(figsize=(8, 6))
        plt.hist(v0_abs, bins=10, alpha=0.5)
        plt.xlabel('Initial Speed (m/s)')
        plt.ylabel('Frequency')
        plt.title('Initial Speed Distribution')

    def plot_t_flight_distribution(self, df):
        ts = []
        for _, row in df.iterrows():
            ts.append(row['tf'])
        plt.figure(figsize=(8, 6))
        plt.hist(ts, bins=10, alpha=0.5)
        plt.xlabel('Max Time (s)')
        plt.ylabel('Frequency')
        plt.title('Max Time Distribution')


    def plot_goal_bins_distribution(self, df, ax = None):
        if ax is None:
            fig, ax = self.field.plot_goal()

        for _, row in df.iterrows():
            g_x_bin = row['g_x_bin']
            g_y_bin = row['g_y_bin']
            x_min = self.field.g_x_grid[g_x_bin - 1] - self.field.GW/2
            x_max = self.field.g_x_grid[g_x_bin] - self.field.GW/2
            y_min = self.field.g_y_grid[g_y_bin - 1]
            y_max = self.field.g_y_grid[g_y_bin]

            ax.fill([x_min, x_max, x_max, x_min],
                    [y_min, y_min, y_max, y_max],
                    color='red', alpha=1/len(df), label=f'Bin ({g_x_bin}, {g_y_bin})')
            
        ax.set_title('Goal Area Bins with Highlighted Landing Bin')
        return ax

    def plot_goal_hits_distribution(self, df, ax = None):
        if ax is None:
            fig, ax = self.field.plot_goal()


        for _, row in df.iterrows():
            g_x = row['g_x']
            g_y = row['g_y']
            g_x_centered = g_x - self.field.W/2
            ax.scatter(g_x_centered, g_y, c='b', s=20, marker='x')
        return ax
    
class Field():
    def __init__(self, W, L, GW, GH):
        self.W = W
        self.L = L
        self.GW = GW
        self.GH = GH
        self.GAW = 3.9  # Goal Area Width
        self.GAL = 0.75  # Goal Area Length
        self.field_corners = np.array([
            [0, 0, 0],
            [W, 0, 0],
            [W, L, 0],
            [0, L, 0]
        ])
        self.goal_corners = np.array([
            [W/2 - GW/2, 0, 0],
            [W/2 + GW/2, 0, 0],
            [W/2 + GW/2, 0, GH],
            [W/2 - GW/2, 0, GH]
        ])
        self.center = np.array([W/2, L/2, 0])

    def plot(self):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        
        field_poly = Poly3DCollection([self.field_corners], alpha=0.3, facecolor='green')
        goal_poly = Poly3DCollection([self.goal_corners], alpha=0.2, facecolor='red')
        ax.add_collection3d(field_poly)
        ax.add_collection3d(goal_poly)
        ax.plot([self.field_corners[0,0], self.field_corners[1,0]], [self.field_corners[0,1], self.field_corners[1,1]], 'k-')
        ax.plot([self.field_corners[1,0], self.field_corners[2,0]], [self.field_corners[1,1], self.field_corners[2,1]], 'k-')
        ax.plot([self.field_corners[2,0], self.field_corners[3,0]], [self.field_corners[2,1], self.field_corners[3,1]], 'k-')
        ax.plot([self.field_corners[3,0], self.field_corners[0,0]], [self.field_corners[3,1], self.field_corners[0,1]], 'k-')

        # Labeling the axes; here the horizontal axes are in meters (landing position)
        # and the vertical axis shows the corresponding v0_z (m/s).
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        self.set_axes_equal(ax)
        return fig, ax
    
    def set_axes_equal(self, ax, Vmax = 10):
            """Sets the aspect ratio of 3D plot to be equal for all axes."""
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            z_limits = ax.get_zlim()

            # Calculate the ranges
            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]
            z_range = z_limits[1] - z_limits[0]

            # Determine the maximum range across all axes
            max_range = max(x_range, y_range, z_range) / 2.0

            # Compute midpoints
            x_middle = np.mean(x_limits)
            y_middle = np.mean(y_limits)
            z_middle = np.mean(z_limits)

            # Set new limits to ensure equal scaling
            ax.set_xlim(x_middle - max_range, x_middle + max_range)
            ax.set_ylim(y_middle - max_range, y_middle + max_range)
            ax.set_zlim(0, 2*(z_middle + max_range))
            # ax.set_zlim(0, 0.5 * Vmax)

    def plot_goal(self, n_goal_bins_x = 5, n_goal_bins_y = 2):
        GH = self.GH
        GW = self.GW

        g_x_grid = np.linspace(0, GW, n_goal_bins_x + 1)
        g_y_grid = np.linspace(0, GH, n_goal_bins_y + 1)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw the outer boundaries of the goal area:
        ax.plot([-GW/2, GW/2], [0, 0], 'k-')       # bottom edge
        ax.plot([-GW/2, GW/2], [GH, GH], 'k-')       # top edge
        ax.plot([-GW/2, -GW/2], [0, GH], 'k-')       # left edge
        ax.plot([GW/2, GW/2], [0, GH], 'k-')         # right edge

        # Draw vertical grid lines:
        for x in g_x_grid:
            x_line = x - GW/2  # shift so that grid runs from -GW/2 to GW/2
            ax.plot([x_line, x_line], [0, GH], 'k--', lw=0.5)

        # Draw horizontal grid lines:
        for y in g_y_grid:
            ax.plot([-GW/2, GW/2], [y, y], 'k--', lw=0.5)
        
        return fig, ax

class Camera():
    def __init__(self, camera_pos, orientation, focal_length, img_width, img_height, pixel_width = 4.86e-6, fps = 30):
        self.camera_pos = camera_pos
        self.orientation = orientation
        self.focal_length = focal_length/pixel_width  # Focal length in pixels
        self.pixel_width = pixel_width
        self.img_width = img_width
        self.img_height = img_height
        self.T = self.look_at(camera_pos, orientation)
        self.T_inv = np.linalg.inv(self.T)
        self.fps = fps
        

    def look_at(self, position, orientation):
        """
        Create a 4x4 transformation matrix that converts world coordinates 
        into camera coordinates, using a camera orientation defined by 
        spherical angles: theta (elevation) and phi (azimuth).
        
        Parameters:
        camera_pos : np.array, shape (3,)
            The camera position in world space.
        theta : float
            The elevation angle in radians (angle above horizontal).
        phi : float
            The azimuth angle in radians (angle in the horizontal plane from the x-axis).
            
        Returns:
        T : np.array, shape (4,4)
            The view (transformation) matrix.
        """
        theta, phi = orientation
        
        # Compute the forward (view) vector from spherical coordinates.
        # Here, we use:
        #   forward = [cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta)]
        forward = np.array([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            np.sin(theta)
        ])
        forward /= np.linalg.norm(forward)  # ensure unit vector

        # For no roll, we choose the "right" vector as the horizontal perpendicular direction.
        # If forward is nearly vertical, choose right arbitrarily.
        if np.isclose(np.cos(theta), 0.0, atol=1e-6):
            right = np.array([1, 0, 0])
        else:
            # A natural choice is:
            right = np.array([np.sin(phi), -np.cos(phi), 0])
            right /= np.linalg.norm(right)

        # The "up" vector is the cross product of forward and right.
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        
        # Build a 3x3 rotation matrix.
        # In camera space, we want:
        #   x-axis = right,  y-axis = up,  and z-axis = -forward.
        R = np.column_stack((right, forward, up))
        
        # Now build a full 4x4 view matrix that includes the translation.
        T = np.eye(4)
        T[:3, :3] = R
        # The translation moves the world so that the camera is at the origin.
        T[3, :3] = -R.T.dot(position)
        return T
    
    @staticmethod
    def get_orientation(camera_pos, camera_target):
        """
        Computes the camera orientation as a unit vector (and the corresponding 
        spherical angles) given a camera position and a target point.
        
        Parameters:
        camera_pos : np.array, shape (3,)
            The camera position in world space.
        camera_target : np.array, shape (3,)
            The point the camera is looking at.
            
        Returns:
        orientation : np.array, shape (3,)
            The unit vector in the direction from camera_pos to camera_target.
        theta : float
            The elevation angle (in radians) computed as arctan2(z, sqrt(x^2+y^2)).
        phi : float
            The azimuth angle (in radians) computed as arctan2(y, x).
        """
        direction = camera_target - camera_pos
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Camera position and target are identical!")
        norm_direction = direction / norm
        # Compute spherical angles. Here we assume:
        #   phi = arctan2(y, x)
        #   theta = arctan2(z, sqrt(x^2+y^2))
        phi = np.arctan2(norm_direction[1], norm_direction[0])
        theta = np.arctan2(norm_direction[2], np.sqrt(norm_direction[0]**2 + norm_direction[1]**2))
        return theta, phi

    def project_point(self, world_point):
        """
        Projects a 3D world point into 2D image pixel coordinates using a simple pinhole camera model.
        Returns (u, v) pixel coordinates and the depth (z in camera coordinates).
        """
        # Transform to camera coordinates:
        world_point_hom = np.hstack([world_point, np.array([1])])  # Convert to homogeneous coordinates
        cam_point = world_point_hom.dot(self.T)
        epsilon = 1e-6
        if cam_point[1] < epsilon:
            # If the point is effectively at or behind the camera, don't project it.
            return None, None
        # Perspective projection: (x, y) -> (f * x/y, f * z/y)
        x_proj = self.focal_length * (cam_point[0] / cam_point[1])
        y_proj = self.focal_length * (cam_point[2] / cam_point[1])
        # print(f"Camera coordinates: {cam_point}, Projected coordinates: ({x_proj}, {y_proj})")
        # Convert to pixel coordinates (origin at bottom-left; center of image is at (img_width/2, img_height/2))
        u = int(self.img_width / 2 + x_proj)
        v = int(self.img_height / 2 - y_proj)  # Invert y-axis for image coordinates
        return np.array([u, v]), cam_point[1]
    
    def project_point_to_world(self, pixel_coords, depth, camera_coords_flag=False):
        """
        Projects 2D pixel coords + depth back to 3D world coords.
        Accepts pixel_coords=(u,v) and depth as either numpy arrays or torch tensors
        (shapes like scalar, [B], [B,chunk], etc).
        """
        u, v = pixel_coords
        # choose backend
        is_torch = isinstance(u, torch.Tensor)

        if is_torch:
            # make sure focal_width etc are floats or torch scalars
            x_norm = (u - self.img_width/2) / self.focal_length
            y_norm = (v - self.img_height/2) / self.focal_length
            ones   = torch.ones_like(depth)
            # build [*,4] hom coords
            cam_h = torch.stack([x_norm * depth,
                                depth,
                                y_norm * depth,
                                ones], dim=-1)           # e.g. [...,4]
            if camera_coords_flag:
                return cam_h[..., :3]
            # apply inverse transform: world = cam_h @ T_invᵀ
            Tinv = self.T_inv  # assume T_inv stored as torch or numpy
            if not isinstance(Tinv, torch.Tensor):
                Tinv = torch.from_numpy(Tinv).to(cam_h.dtype).to(cam_h.device)
            world_h = cam_h.matmul(Tinv)                       # [...,4]
            return world_h[..., :3]

        else:
            # numpy path
            x_norm = (u - self.img_width/2) / self.focal_length
            y_norm = (v - self.img_height/2) / self.focal_length
            ones   = np.ones_like(depth)
            cam_h  = np.stack([x_norm*depth,
                            depth,
                            y_norm*depth,
                            ones], axis=-1)         # [...,4]
            if camera_coords_flag:
                return cam_h[..., :3]
            world_h = cam_h.dot(self.T_inv)                    # [...,4]
            return world_h[..., :3]


    def project_ball_camera_to_world(self, camera_coords, camera_coords_flag=False):
        """
        camera_coords = (x_cam, y_cam, R_cam)
        where each can be scalar, numpy array, or torch tensor of shape [B,chunk]
        """
        x_cam, y_cam, R_cam = camera_coords
        # get_depth must similarly support arrays or tensors
        depth = self.get_depth(BALL_RADIUS, R_cam)
        return self.project_point_to_world((x_cam, y_cam),
                                        depth,
                                        camera_coords_flag=camera_coords_flag)

    def create_frames(self, trajectory, ball_radius_world, save_frames=True):
        # Create a directory to store frames.
        if save_frames:
            frames_dir = "frames"
            os.makedirs(frames_dir, exist_ok=True)

        imgs = []
        # Loop over each time step, project the ball's center, and draw a circle on a blank image.
        for i, pos in enumerate(trajectory):
            # Create a blank white image.
            img = np.ones((self.img_height, self.img_width, 3), dtype=np.uint8) * 255

            # Project the ball's center to image coordinates.
            proj, depth = self.project_point(pos)
            if proj is None:
                # If the point is behind the camera, skip drawing.
                print(f"Skipping frame {i}: Point is behind the camera.")
                continue
            u, v = proj
            print(f"Frame {i}: Projected coordinates: ({u}, {v}), Depth: {depth}")

            # Compute the projected radius in pixels.
            # In a pinhole model, projected size ~ focal_length * (ball_radius_world / depth)
            radius_pixels = self.get_radius_pixels(ball_radius_world, depth)
            
            # Draw the ball as a filled circle (using OpenCV).
            cv2.circle(img, (u, v), radius_pixels, (0, 0, 255), -1)

            # img = cv2.flip(img, 0)
            
            imgs.append(img.copy())
            # Optionally, add frame number text.
            img_file = cv2.putText(img, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Save the frame image.
            if save_frames:
                filename = os.path.join(frames_dir, f"frame_{i:03d}.png")
                cv2.imwrite(filename, img_file)
            
        if save_frames: print("Frames saved in directory:", frames_dir)
        return imgs
    
    def get_radius_pixels(self, ball_radius_world, depth):
        """
        Given the ball radius in world coordinates and the depth of the ball in camera coordinates,
        compute the projected radius in pixels.
        """
        return int(self.focal_length * (ball_radius_world / depth))
    
    def get_depth(self, ball_radius_world, radius_pixels):
        """
        Given the ball radius in world coordinates and the projected radius in pixels,
        compute the depth of the ball in camera coordinates.
        """
        return self.focal_length * (ball_radius_world / radius_pixels)

    def create_video(self, frames, fps=None, filename="trajectory.mp4"):
        if fps is None:
            fps = self.fps
        # Create a video from the frames using OpenCV.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try using MJPG codec
        out = cv2.VideoWriter(filename, fourcc, fps, (self.img_width, self.img_height))
        if not out.isOpened():
            print("Error: VideoWriter not opened")
            return
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)  # Ensure frame is uint8
            out.write(frame)
        out.release()
        # imageio.mimsave('video_output.mp4', frames, fps=fps)
        print(f"Video saved as {filename}")
        
    def plot(self, ax):
        ax.scatter(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2], c='red', label='Camera Position')
        ax.plot([self.camera_pos[0], self.camera_pos[0]], [self.camera_pos[1], self.camera_pos[1]], [0, self.camera_pos[2]], 'k--')

    def is_point_in_field_of_view(self, r):
        field_of_view_constraint = lambda r: (
            r[0] > 0 and 
            r[0] < self.img_width and
            r[1] > 0 and
            r[1] < self.img_height
        )
        r_proj = self.project_point(r)[0]
        if r_proj is None:
            return False
        return field_of_view_constraint(r_proj)
    
    def calculate_pixel_real_world_size(self, distance):
        """
        Calculate the real-world size represented by one pixel in a pinhole camera model.

        Parameters:
        sensor_width (float): Physical width of the camera sensor in meters.
        image_width (int): Number of pixels in the image width.
        focal_length (float): Focal length of the camera in meters.
        distance (float): Distance to the object in meters.

        Returns:
        float: Real-world size per pixel in meters.
        """
        real_size_per_pixel = self.pixel_width * (distance / self.focal_length)
        return real_size_per_pixel
