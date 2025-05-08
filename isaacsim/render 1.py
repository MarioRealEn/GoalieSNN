import omni
from omni.isaac.core import World
import numpy as np
import time
import os
from PIL import Image
from pxr import UsdGeom, Gf
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

# Get the stage
stage = omni.usd.get_context().get_stage()

# Retrieve the existing robot
robot_prim_path = "/World/Robot5"
robot_prim = stage.GetPrimAtPath(robot_prim_path)

# Retrieve the existing camera under the robot
camera_prim_path = "/World/Robot5/ZED_X/base_link/ZED_X/CameraLeft"

# Create output directory
output_dir = "C:\\Work\\robosoccer\\simulator\\images"
os.makedirs(output_dir, exist_ok=True)
           

def update_prim_translation(prim, translation):
    """Update the translation of the robot."""
    xform = UsdGeom.Xformable(prim)
    if xform:
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3f(translation))
                return
            else:
                print("No translation found")

def update_prim_rotation(prim, rotation_z):
    """Update the rotation of the robot on Z axis."""
    xform = UsdGeom.Xformable(prim)
    if xform:
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:  # Correct type for rotation
                op.Set(Gf.Vec3f(0.0, 0.0, float(rotation_z)))  # Convert rotation_z to float
            else:
                print("No rotation found")

def move_robot_and_capture():
    for x in range(-7, 8, 8):
        for y in range(-11, 12, 8):
            for rot in range(0, 181, 180):
                print(f"Moving robot to position x: {x}, y: {y} with rotation: {rot} degrees")
                update_prim_translation(robot_prim, (float(x), float(y), 0.0))
                #update_prim_rotation(robot_prim, float(rot))
                time.sleep(0.5)  # Wait for visualization
                vp_api = get_active_viewport()
                image_path = os.path.join(output_dir, f"image_{x}_{y}_{rot}.png")
                capture_viewport_to_file(vp_api, image_path)
                print(f"Captured image to {image_path}")
                time.sleep(0.5)  # Wait for visualization

move_robot_and_capture()