import omni
import numpy as np
import os
import asyncio
import csv
import sys
import pandas as pd
# print(sys.path)
from omni.isaac.core import World
from pxr import UsdGeom, Gf
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.widget.viewport.capture import FileCapture
import carb.settings

async def set_antialiasing_to_fxaa():
    # Wait a moment to ensure rendering settings are ready
    await viewport_api.wait_for_rendered_frames(5)  # let the viewport settle

    settings = carb.settings.get_settings()
    # Print all RTX-related settings
    print("AA default mode:", settings.get("/rtx/post/aa/op"))
  
    settings.set("/rtx/post/aa/op", 2)

    # Confirm
    settings = carb.settings.get_settings()
    print("AA mode set to:", settings.get("/rtx/post/aa/op"))

# Schedule the change
# asyncio.ensure_future(set_antialiasing_to_fxaa())

# Get the stage
stage = omni.usd.get_context().get_stage()

# Retrieve the existing robot
robot_prim_path = "/World/Robot5"
robot_prim = stage.GetPrimAtPath(robot_prim_path)

# Ball
ball_prim_path = "/World/Ball"
ball_prim = stage.GetPrimAtPath(ball_prim_path)


# Create output directory
output_dir = "C:\\Work\\robosoccer\\simulator\\images"
os.makedirs(output_dir, exist_ok=True)

# Setup viewport preview
viewport_api = get_active_viewport()


# Set camera path
viewport_api.camera_path = "/World/Robot5/ZED_X/base_link/ZED_X/CameraLeft"


def update_prim_translation(prim, translation):
    """Update or create a translation transform for the robot."""
    xform = UsdGeom.Xformable(prim)
    if xform:
        xform_ops = xform.GetOrderedXformOps()
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3f(translation))
                return
        
        # If no translate op found, add a new one
        xform.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3f(translation))
    else:
        print("Failed to get Xformable for the robot.")

def update_prim_rotation(prim, rotation_z):
    """Update the rotation of the robot on Z axis."""
    xform = UsdGeom.Xformable(prim)
    if xform:
        xform_ops = xform.GetOrderedXformOps()
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                op.Set(Gf.Vec3f(0.0, 0.0, float(rotation_z)))
                return
        
        # If no rotation op found, add a new one
        xform.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ).Set(Gf.Vec3f(0.0, 0.0, float(rotation_z)))
    else:
        print("Failed to get Xformable for the robot.")


async def move_robot_and_capture():
    for x in range(-7, 8, 3):
        for y in range(-11, 12, 3):
            for rot in range(0, 181, 90):
                print(f"Moving robot to position x: {x}, y: {y} with rotation: {rot} degrees")

                # Update translation and force stage update
                update_prim_translation(robot_prim, (x, y, 0.0))
                update_prim_rotation(robot_prim, rot)
                stage.Flatten()  # Ensure the stage is updated

                # Wait for render settings to be updated
                await viewport_api.wait_for_render_settings_change()
                await viewport_api.wait_for_rendered_frames(1)

                # Capture the image
                image_path = os.path.join(output_dir, f"image_{x}_{y}_{rot}.png")
                capture = viewport_api.schedule_capture(FileCapture(image_path))
                captured_aovs = await capture.wait_for_result()

                if captured_aovs:
                    print(f'AOV "{captured_aovs[0]}" was written to "{image_path}"')
                else:
                    print(f'No image was written to "{image_path}"')

async def move_ball_and_capture(trajectory, id, output_dir):
    for idx, position in enumerate(trajectory):
        x, y, z = position
        x -= 7
        y -= 11
        print(f"Moving ball to position x: {x}, y: {y}, z: {z}")

        # Update translation and force stage update
        update_prim_translation(ball_prim, (x, y, z))
        stage.Flatten()  # Ensure the stage is updated

        # Wait for render settings to be updated
        await viewport_api.wait_for_render_settings_change()
        await viewport_api.wait_for_rendered_frames(2)
        await asyncio.sleep(0.1)

        # Capture the image
        image_path = os.path.join(output_dir, id, f"frame_{idx:03}.png")
        capture = viewport_api.schedule_capture(FileCapture(image_path))
        captured_aovs = await capture.wait_for_result()

        if captured_aovs:
            print(f'AOV "{captured_aovs[0]}" was written to "{image_path}"')
        else:
            print(f'No image was written to "{image_path}"')

async def generate_images(all_init_conditions, trajectory_gen):
    for _, init_conditions in all_init_conditions.iterrows():
        # Unpack the initial conditions
        r0 = np.zeros(3)
        v0 = np.zeros(3)
        g_bins = np.zeros(2)
        id, r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], g_x, g_y, g_bins[0], g_bins[1], t_flight, is_roll = init_conditions
        trajectory = trajectory_gen.sample_trajectory(r0, v0, t_flight, roll = is_roll)

        # Move the ball and capture images
        await move_ball_and_capture(trajectory, id)

print("Starting the simulation...")
csv_trajectories = r"C:\Users\realenriquem\OneDrive - Sioux Group B.V\Documents\Git\BallTrackingSNN\dataset_python_sim\trajectories.csv"
df_trajectories = pd.read_csv(csv_trajectories, header=0)
csv_positions = r"C:\Users\realenriquem\OneDrive - Sioux Group B.V\Documents\Git\BallTrackingSNN\dataset_python_sim\positions.csv"
df_positions = pd.read_csv(csv_positions, header=0)
print("Loaded trajectories and positions from CSV files.")
positions = df_positions.to_numpy()
positions = [[float(coord) for coord in position] for position in positions]
print("Converted positions to float.")
cwd = os.getcwd()
output_dir = os.path.join(cwd, "frames")
cont = 0
cont_max = 3
for trajectory in df_trajectories:
    traj_id = trajectory[0]
    print(f"Trajectory {traj_id}")
    pos_traj = df_positions[df_positions['tr'] == traj_id].to_numpy()
    # Print 10 first elements of the trajectory
    print(pos_traj[:10])
    asyncio.create_task(move_ball_and_capture(pos_traj, traj_id, output_dir))
    cont += 1
    if cont >= cont_max:
        break

# async def funct():
#     await move_ball_and_capture(positions, "who_knows_example")
#     await set_antialiasing_to_fxaa()

# asyncio.create_task(funct())