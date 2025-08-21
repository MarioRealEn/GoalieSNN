import cv2
import shutil
import numpy as np
import pandas as pd
import os
import glob


# def track_ball(video_path, output_dir, initial_position, initial_radius):
#     # Load video
#     cap = cv2.VideoCapture(video_path)
#     # Ensure output directory exists for frames
#     tracked_frame_output_dir = "tracked_frames_circle/"
#     os.makedirs(tracked_frame_output_dir, exist_ok=True)

#     # CLAHE for contrast
#     # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

#     # Kalman initialization
#     kalman = cv2.KalmanFilter(6, 3)  # states: x,y,r,dx,dy,dr; measurements: x,y,r
#     kalman.transitionMatrix = np.array([
#         [1, 0, 0, 1, 0, 0],
#         [0, 1, 0, 0, 1, 0],
#         [0, 0, 1, 0, 0, 1],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 1]
#     ], np.float32)
#     kalman.measurementMatrix = np.array([
#         [1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0]
#     ], np.float32)
#     kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
#     kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5

#     # You can set an approximate initial state for the filter:
#     initial_position = (710, 266)
#     initial_radius   = 15
#     kalman.statePre = np.array([
#         [initial_position[0]],
#         [initial_position[1]],
#         [initial_radius],
#         [0],
#         [0],
#         [0]
#     ], np.float32)

#     # Distance threshold for “teleport” checks
#     TELEPORT_THRESHOLD = 100.0

#     tracked_positions = []
#     frame_count = 0

#     # We'll track whether we've accepted any detection yet
#     initialized = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Kalman predict
#         predicted = kalman.predict()
#         pred_x, pred_y, pred_r = int(predicted[0]), int(predicted[1]), int(predicted[2])

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Optionally apply CLAHE:
#         # gray = clahe.apply(gray)

#         circles = cv2.HoughCircles(
#             gray,
#             cv2.HOUGH_GRADIENT,
#             dp=1.2,
#             minDist=30,
#             param1=50,
#             param2=30,
#             minRadius=5,
#             maxRadius=60
#         )

#         circle_chosen = None
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             # Among all circles, pick the one closest to Kalman prediction
#             best_circle = min(
#                 circles[0, :],
#                 key=lambda c: np.linalg.norm([c[0] - pred_x, c[1] - pred_y])
#             )
#             x_detect, y_detect, r_detect = best_circle

#             # Check “teleport” only if already initialized
#             if not initialized:
#                 # If we haven't accepted any detection yet, accept the first valid circle outright
#                 circle_chosen = (x_detect, y_detect, r_detect)
#                 initialized = True
#             else:
#                 # Once initialized, apply the teleport threshold
#                 dist_from_pred = np.linalg.norm([(x_detect - pred_x), (y_detect - pred_y)])
#                 if dist_from_pred < TELEPORT_THRESHOLD:
#                     circle_chosen = (x_detect, y_detect, r_detect)

#         # Update Kalman + draw if we have a chosen circle
#         if circle_chosen is not None:
#             x, y, r = circle_chosen
#             measurement = np.array([[np.float32(x)],
#                                     [np.float32(y)],
#                                     [np.float32(r)]], np.float32)
#             kalman.correct(measurement)

#             # Draw
#             cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
#             cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

#             # Save data
#             tracked_positions.append([frame_count, x, y, r])

#         # Save frame with or without circle
#         outpath = os.path.join(tracked_frame_output_dir, f"frame_{frame_count:04d}.png")
#         cv2.imwrite(outpath, frame)
#         frame_count += 1

#     cap.release()

#     # Convert to DataFrame and save
#     df = pd.DataFrame(tracked_positions, columns=["Frame", "X", "Y", "Radius"])
#     df.to_csv("circle_tracked_ball_positions.csv", index=False)

#     # Encode frames into a video
#     os.system(f"ffmpeg -framerate 100 -i {tracked_frame_output_dir}/frame_%04d.png "
#             f"-c:v libx264 -pix_fmt yuv420p {output_dir}/circle_tracked_ball_video.mp4")

#     print("Tracking completed.")


def track_ball(video_path, output_dir, teleport_threshold=100.0):
    """
    Tracks a ball in a video without needing an explicit initial position.
    The first valid Hough Circle detection initializes the Kalman filter.
    
    :param video_path: Path to the input video.
    :param output_dir: Directory to store output frames and final tracked video.
    :param teleport_threshold: Distance threshold for discarding "teleport" frames.
    """
    # --- Prepare video input and output directories ---
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    tracked_frame_output_dir = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_frames"))
    if os.path.exists(tracked_frame_output_dir):
        for file in os.listdir(tracked_frame_output_dir):
            file_path = os.path.join(tracked_frame_output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove subdirectory
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    os.makedirs(tracked_frame_output_dir, exist_ok=True)

    # --- Create a Kalman Filter for (x, y, r, dx, dy, dr) with measurements (x, y, r) ---
    kalman = cv2.KalmanFilter(6, 3)

    # State transition model:
    # [ x ]      [1 0 0 1 0 0]
    # [ y ]      [0 1 0 0 1 0]
    # [ r ]   =  [0 0 1 0 0 1] * previous_state
    # [dx ]      [0 0 0 1 0 0]
    # [dy ]      [0 0 0 0 1 0]
    # [dr ]      [0 0 0 0 0 1]
    kalman.transitionMatrix = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)

    # Measurement model: we directly measure (x, y, r) from HoughCircles
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], dtype=np.float32)

    # Tuning parameters
    kalman.processNoiseCov  = np.eye(6, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5

    # We won't set kalman.statePre yet, since we have no idea where the ball is.
    # We'll do it after the first successful detection.

    tracked_positions = []
    frame_count = 0
    initialized = False  # Whether the Kalman filter is yet initialized with a real detection
    last_detection = None  # Last accepted detection (x, y, r)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape

        # If we haven't initialized yet, no point in predicting,
        # but let's do it anyway to keep the flow consistent.
        predicted = kalman.predict()
        pred_x, pred_y, pred_r = predicted[0,0], predicted[1,0], predicted[2,0]

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not initialized:
            minRadius = 5
            maxRadius = 100
        else:
            # If we have already initialized, we can use the prediction to narrow down the search
            minRadius = int(max(5, pred_r - 10))
            maxRadius = int(min(100, pred_r + 10))

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=40,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        circle_chosen = None
        def cost(circle, pred, last_detection, dt=1):
            # circle: (x, y, r)
            # pred: predicted state [pred_x, pred_y, pred_r, dx, dy, dr]
            # last_detection: previous accepted measurement [x_prev, y_prev, r_prev]
            x, y, r = circle
            pred_x, pred_y, pred_r, dx_pred, dy_pred, dr_pred = pred
            
            # Compute measured derivative (velocity) from last detection:
            if last_detection is None:
                measured_dx, measured_dy, measured_dr = dx_pred, dy_pred, dr_pred
            else:
                x_prev, y_prev, r_prev = last_detection
                measured_dx = (x - x_prev) / dt
                measured_dy = (y - y_prev) / dt
                measured_dr = (r - r_prev) / dt
            
            # Calculate differences
            diff_position = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
            diff_velocity = np.sqrt((measured_dx - dx_pred)**2 + (measured_dy - dy_pred)**2)
            diff_radius = abs(r - pred_r)
            diff_dr = abs(measured_dr - dr_pred)
            
            # Combine them using weights (adjust as needed)
            cost_value = diff_position + diff_velocity + diff_radius + diff_dr
            return cost_value

        if circles is not None:
            # print(len(circles[0]))
            circles = np.uint16(np.around(circles))
            # Among all circles, pick the one closest to the predicted state

            best_circle = min(
                circles[0,:],
                key=lambda c: cost(c, predicted, last_detection)
            )
            best_circle = min(
                circles[0, :],
                key=lambda c: np.linalg.norm([c[0] - pred_x, c[1] - pred_y])
            )
            x_detect, y_detect, r_detect = best_circle
            dist_from_pred = np.linalg.norm([x_detect - pred_x, y_detect - pred_y])

            if not initialized:
                # The very first detection we accept unconditionally
                # and use it to initialize the Kalman filter state.
                circle_chosen = (x_detect, y_detect, r_detect)
                # We can directly set the pre-state to that detection:
                kalman.statePre = np.array([
                    [float(x_detect)],
                    [float(y_detect)],
                    [float(r_detect)],
                    [0.0],
                    [0.0],
                    [0.0]
                ], dtype=np.float32)
                initialized = True
            else:
                # # We are already initialized, so check the teleport threshold
                # if dist_from_pred < teleport_threshold:
                circle_chosen = (x_detect, y_detect, r_detect)

        # If we have a chosen circle, correct the Kalman filter and draw
        if circle_chosen is not None:
            last_detection = circle_chosen
            x, y, r = circle_chosen
            measurement = np.array([[float(x)], [float(y)], [float(r)]], np.float32)
            kalman.correct(measurement)

            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # Store for CSV
            video_id = video_name.replace(".mp4", "").split("_")[1:]
            video_id = "_".join(video_id)
            frame_id = f"{video_id}_{frame_count:04d}"
            y= height - y # Flip y-coordinate because I want the origin to be at the bottom left
            tracked_positions.append([frame_id, x, y, r])

        # Save output frame
        outpath = os.path.join(tracked_frame_output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(outpath, frame)

        frame_count += 1

    cap.release()

    # Save to CSV
    df = pd.DataFrame(tracked_positions, columns=["Frame", "X", "Y", "Radius"])
    csv_path = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_circle.csv"))
    df.to_csv(csv_path, index=False)

    # Encode frames into a video
    output_video_path = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_circle.mp4"))
    cmd = (
        f"ffmpeg -framerate 100 -i {tracked_frame_output_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p {output_video_path}"
    )
    os.system(cmd)

    print(f"Tracking completed. CSV saved at: {csv_path}")
    print(f"Tracked video saved at: {output_video_path}")



def choose_stable_circle(detections, dist_thresh=20):
    """
    Groups circles that are close to each other (within dist_thresh).
    Returns the average (x, y, r) of the largest group (the cluster with most circles).
    """
    clusters = []
    for c in detections:
        placed = False
        for cluster in clusters:
            if np.linalg.norm([c[0] - cluster[0][0], c[1] - cluster[0][1]]) < dist_thresh:
                cluster.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])

    best_cluster = max(clusters, key=len)
    xs = [cc[0] for cc in best_cluster]
    ys = [cc[1] for cc in best_cluster]
    rs = [cc[2] for cc in best_cluster]
    return (int(np.mean(xs)), int(np.mean(ys)), int(np.mean(rs)))

def track_ball2(video_path, output_dir, teleport_threshold=100.0):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    tracked_frame_output_dir = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_frames"))
    if os.path.exists(tracked_frame_output_dir):
        for file in os.listdir(tracked_frame_output_dir):
            file_path = os.path.join(tracked_frame_output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    os.makedirs(tracked_frame_output_dir, exist_ok=True)

    kalman = cv2.KalmanFilter(6, 3)
    kalman.transitionMatrix = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], dtype=np.float32)
    kalman.processNoiseCov  = np.eye(6, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5

    initialized = False
    last_detection = None
    tracked_positions = []
    frame_count = 0

    # Instead of waiting for 10 frames total, wait for 10 frames that actually had circles.
    NUM_CIRCLE_FRAMES = 10
    first_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape

        predicted = kalman.predict()
        pred_x, pred_y, pred_r = predicted[0,0], predicted[1,0], predicted[2,0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        minRadius = 5
        maxRadius = 100
        # if not initialized:
        #     minRadius = 5
        #     maxRadius = 100
        # else:
        #     minRadius = int(max(5, pred_r - 20))
        #     maxRadius = int(min(100, pred_r + 20))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=40,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        circle_chosen = None
        if circles is not None:
            print(f"Frame {frame_count} circle detected")
            print(len(circles[0]))
            circles = np.uint16(np.around(circles))
            # pick the circle closest to predicted
            best_circle = min(
                circles[0, :],
                key=lambda c: np.linalg.norm([c[0] - pred_x, c[1] - pred_y, c[2] - pred_r])
            )
            x_detect, y_detect, r_detect = best_circle
            dist_from_pred = np.linalg.norm([x_detect - pred_x, y_detect - pred_y])

            if not initialized:
                # Accumulate these circle frames until we have enough
                first_detections.append((x_detect, y_detect, r_detect))
                if len(first_detections) >= NUM_CIRCLE_FRAMES:
                    stable_x, stable_y, stable_r = choose_stable_circle(first_detections)
                    print(f"Stable circle in frame {frame_count}: {stable_x}, {stable_y}, {stable_r}")
                    kalman.statePre = np.array([
                        [float(stable_x)],
                        [float(stable_y)],
                        [float(stable_r)],
                        [0.0],
                        [0.0],
                        [0.0]
                    ], dtype=np.float32)
                    initialized = True
                    circle_chosen = (stable_x, stable_y, stable_r)
            else:
                if dist_from_pred < teleport_threshold:
                    circle_chosen = (x_detect, y_detect, r_detect)
                else:
                    print(f"Teleport detected in frame {frame_count}")

        if circle_chosen is not None:
            print(f"Frame {frame_count} circle chosen")
            # last_detection = circle_chosen
            x, y, r = circle_chosen
            measurement = np.array([[float(x)], [float(y)], [float(r)]], np.float32)
            kalman.correct(measurement)

            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            frame_id = f"{video_name}_{frame_count:04d}"
            y_flip = height - y
            tracked_positions.append([frame_id, x, y_flip, r])

        outpath = os.path.join(tracked_frame_output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(outpath, frame)
        frame_count += 1

    cap.release()
    df = pd.DataFrame(tracked_positions, columns=["Frame", "X", "Y", "Radius"])
    csv_path = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_circle.csv"))
    df.to_csv(csv_path, index=False)

    output_video_path = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_circle.mp4"))
    cmd = (
        f"ffmpeg -framerate 100 -i {tracked_frame_output_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p {output_video_path}"
    )
    os.system(cmd)

    print(f"Tracking completed. CSV saved at: {csv_path}")
    print(f"Tracked video saved at: {output_video_path}")

def choose_stable_circle_with_velocity(detections, pos_thresh=20, velocity_variation_thresh=5, angle_variation_thresh=0.2):
    """
    Groups detections that are close in position (within pos_thresh) and checks if the
    velocities between consecutive detections are stable in both magnitude and direction.
    
    Detections is a list of tuples: (frame, x, y, r).
    
    - velocity_variation_thresh: maximum allowed standard deviation in speed (pixels/frame).
    - angle_variation_thresh: maximum allowed standard deviation (in radians) for velocity angles.
    
    Returns the average (x, y, r) of the largest cluster if both the speeds and the
    velocity directions are stable; otherwise returns None.
    """

    # Group detections based on spatial closeness
    clusters = []
    for d in detections:
        placed = False
        for cluster in clusters:
            if np.linalg.norm([d[1] - cluster[0][1], d[2] - cluster[0][2]]) < pos_thresh:
                cluster.append(d)
                placed = True
                break
        if not placed:
            clusters.append([d])

    if not clusters:
        return None

    best_cluster = max(clusters, key=len)
    if len(best_cluster) < 2:
        return None  # Need at least two detections to compute velocity

    # Sort the cluster by frame number
    best_cluster.sort(key=lambda d: d[0])
    
    speeds = []
    angles = []
    
    # Compute velocities (difference between consecutive detections)
    for i in range(1, len(best_cluster)):
        frame_prev, x_prev, y_prev, _ = best_cluster[i-1]
        frame_curr, x_curr, y_curr, _ = best_cluster[i]
        dt = frame_curr - frame_prev
        if dt == 0:
            continue
        dx = x_curr - x_prev
        dy = y_curr - y_prev
        speed = np.sqrt(dx*dx + dy*dy) / dt
        angle = np.arctan2(dy, dx)
        speeds.append(speed)
        angles.append(angle)
    
    if not speeds or not angles:
        return None
    
    # Unwrap angles to avoid discontinuities at the ±π boundary
    angles = np.unwrap(angles)
    speed_std = np.std(speeds)
    angle_std = np.std(angles)
    
    if speed_std > velocity_variation_thresh or angle_std > angle_variation_thresh:
        return None  # The detections are too unstable
    
    # If stable, return the average circle (position and radius) from the cluster
    xs = [d[1] for d in best_cluster]
    ys = [d[2] for d in best_cluster]
    rs = [d[3] for d in best_cluster]
    return (int(np.mean(xs)), int(np.mean(ys)), int(np.mean(rs)))


def track_ball3(video_path, output_dir, teleport_threshold=100.0):

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    tracked_frame_output_dir = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_frames"))
    if os.path.exists(tracked_frame_output_dir):
        for file in os.listdir(tracked_frame_output_dir):
            file_path = os.path.join(tracked_frame_output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    os.makedirs(tracked_frame_output_dir, exist_ok=True)

    kalman = cv2.KalmanFilter(6, 3)
    kalman.transitionMatrix = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], dtype=np.float32)
    kalman.processNoiseCov  = np.eye(6, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5

    initialized = False
    tracked_positions = []
    frame_count = 0

    # Accumulate circle detections as tuples: (frame, x, y, r)
    NUM_CIRCLE_FRAMES = 10
    first_detections = []
    miss_count = 0  # Counts consecutive frames with no circle detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape

        predicted = kalman.predict()
        pred_x, pred_y, pred_r = predicted[0, 0], predicted[1, 0], predicted[2, 0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        minRadius = 5
        maxRadius = 100
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=40,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        circle_chosen = None
        if circles is not None:
            print(f"Frame {frame_count} circle detected, count: {len(circles[0])}")
            circles = np.uint16(np.around(circles))
            # Choose the circle closest to the predicted state (including radius)
            best_circle = min(
                circles[0, :],
                key=lambda c: np.linalg.norm([c[0] - pred_x, c[1] - pred_y, c[2] - pred_r])
            )
            x_detect, y_detect, r_detect = best_circle
            dist_from_pred = np.linalg.norm([x_detect - pred_x, y_detect - pred_y])

            if not initialized:
                # Accumulate detections (with frame number) to decide on stability.
                first_detections.append((frame_count, x_detect, y_detect, r_detect))
                if len(first_detections) >= NUM_CIRCLE_FRAMES:
                    stable_circle = choose_stable_circle_with_velocity(first_detections)
                    if stable_circle is not None:
                        stable_x, stable_y, stable_r = stable_circle
                        print(f"Stable circle found at frame {frame_count}: {stable_x}, {stable_y}, {stable_r}")
                        kalman.statePre = np.array([
                            [float(stable_x)],
                            [float(stable_y)],
                            [float(stable_r)],
                            [0.0],
                            [0.0],
                            [0.0]
                        ], dtype=np.float32)
                        initialized = True
                        circle_chosen = (stable_x, stable_y, stable_r)
                        miss_count = 0  # Reset miss count when detection is accepted
                    else:
                        print(f"Not stable yet at frame {frame_count}; waiting for more consistent detections...")
            else:
                if dist_from_pred < teleport_threshold:
                    circle_chosen = (x_detect, y_detect, r_detect)
                else:
                    print(f"Teleport detected in frame {frame_count}")
        else:
            print(f"Frame {frame_count}: no circle detected.")

        # If no circle was chosen in this frame, increase the miss count.
        if circle_chosen is None:
            miss_count += 1
            # If no circle detection for 5 consecutive frames, reset the Kalman filter.
            if miss_count >= 5:
                print(f"No circle detected for {miss_count} consecutive frames; resetting Kalman filter.")
                initialized = False
                first_detections = []
                miss_count = 0  # Reset miss count after resetting
        else:
            # If a circle was chosen, reset the miss counter.
            miss_count = 0

        if circle_chosen is not None:
            print(f"Frame {frame_count} circle chosen")
            x, y, r = circle_chosen
            measurement = np.array([[float(x)], [float(y)], [float(r)]], np.float32)
            kalman.correct(measurement)

            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            frame_id = f"{video_name}_{frame_count:04d}"
            y_flip = height - y
            tracked_positions.append([frame_id, x, y_flip, r])

        outpath = os.path.join(tracked_frame_output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(outpath, frame)
        frame_count += 1

    cap.release()
    df = pd.DataFrame(tracked_positions, columns=["Frame", "X", "Y", "Radius"])
    csv_path = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_circle.csv"))
    df.to_csv(csv_path, index=False)

    output_video_path = os.path.join(output_dir, video_name.replace(".mp4", "_tracked_circle.mp4"))
    cmd = (
        f"ffmpeg -framerate 100 -i {tracked_frame_output_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p {output_video_path}"
    )
    os.system(cmd)

    print(f"Tracking completed. CSV saved at: {csv_path}")
    print(f"Tracked video saved at: {output_video_path}")

if __name__ == "__main__":
    # Path to the video file
    video_path = "real100fps.mp4"
    output_dir = "output_circle"
    track_ball(video_path, output_dir)

def create_merged_dataset(csv_dir, videos_dir, dataset_dir):
    """
    Merges individual tracked CSVs into a global labels CSV and extracts original frames
    (without the tracking overlay) into a dataset folder.

    Assumptions:
      - Each CSV is named like "videoID_tracked_circle.csv".
      - The corresponding original video is named "videoID.mp4" and is located in videos_dir.
      - The CSV's "Frame" column is either a number or a string formatted as "videoID_XXXX", where XXXX is the frame index.

    Parameters:
        csv_dir (str): Directory containing the tracked CSV files.
        videos_dir (str): Directory containing the original video files.
        dataset_dir (str): Output folder where original frames will be saved.
        global_csv_path (str): Filepath to save the merged global CSV.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    frames_dir = os.path.join(dataset_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Find all tracked CSV files (e.g., "*_tracked_circle.csv")
    csv_files = glob.glob(os.path.join(csv_dir, '**', "*_tracked_circle.csv"), recursive=True)
    global_labels = []  # To store rows for the merged CSV
    global_frame_index = 0  # Global frame counter for unique naming

    for csv_file in csv_files:
        # Derive the video id from the CSV filename:
        # e.g., if csv_file is "real100fps_tracked_circle.csv", then video_id is "real100fps"
        base_name = os.path.basename(csv_file)
        video_id = base_name.replace("_tracked_circle.csv", "")
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(videos_dir, video_filename)
        
        if not os.path.exists(video_path):
            print(f"Video file {video_path} not found. Skipping {csv_file}.")
            continue
        
        # Read the CSV for this video
        df = pd.read_csv(csv_file)
        
        # Open the original video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video file {video_path}.")
            continue

        for _, row in df.iterrows():
            # Extract the frame index from the "Frame" column.
            # If the CSV "Frame" value is like "videoID_0001", split and take the numeric part.
            frame_id = str(row["Frame"])
            if "_" in frame_id:
                try:
                    frame_number = int(frame_id.split("_")[-1])
                except ValueError:
                    print(f"Invalid frame number '{frame_id}' in {csv_file}. Skipping row.")
                    continue
            else:
                frame_number = int(frame_id)
            
            # Set the video to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_id}.")
                continue
            
            # Save the original (unannotated) frame using a global index naming scheme
            frame_filename = f"{frame_id}.png"
            output_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(output_path, frame)
            
            # Append the row to the global labels.
            # The new "Frame" column will hold the global frame index,
            # while X, Y, and Radius come from the original CSV.
            global_labels.append({
                "Frame": frame_id,
                "X": row["X"],
                "Y": row["Y"],
                "Radius": row["Radius"]
            })
            global_frame_index += 1

        cap.release()

    # Create a DataFrame for all merged labels and save it.
    global_df = pd.DataFrame(global_labels, columns=["Frame", "X", "Y", "Radius"])
    global_csv_path = os.path.join(dataset_dir, 'labels.csv')
    global_df.to_csv(global_csv_path, index=False)
    print(f"Global labels CSV saved at: {global_csv_path}")
    print(f"Dataset frames saved in: {frames_dir}")