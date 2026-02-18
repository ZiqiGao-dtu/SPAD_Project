import numpy as np
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import cv2
import pandas as pd
from io import StringIO
import json
from tqdm import tqdm

# ### Intrinsics of spad ###
fx, fy = 514.9963, 515.7036
cx, cy = 247.3648, 249.7464
s = 0
K = np.array([
    [fx, s,  cx],
    [0,  fy, cy],
    [0,   0,  1]
])
# Distortion coefficients in OpenCV format: [k1, k2, p1, p2, k3], Radial distortion: [k1, k2, k3], Tangential distortion: [p1, p2]
dist_coeffs = np.array([-0.1397, 0.2914, 0.0, 0.0, 0.0], dtype=np.float32)


# ### Load all image names ###
spad_images_directory = r'spad\scaled4bit'
all_images = sorted([f for f in os.listdir(spad_images_directory) if f.endswith('.png')])  # Sorted list of JPG file names


# ### Read data from .csv and .json ###
csv_timestamp = "Take 2024-09-20 11.50.48 AM"
optitrack_csv = f"{csv_timestamp}.csv"
optitrack_json = f"annotations/{csv_timestamp}_optitrack.json"
spad_json   = f"annotations/{csv_timestamp}_spad.json"


# Read from optitrack_csv
with open(optitrack_csv, "r") as f:
    lines = f.readlines()
metadata = [item for line in lines[:6] for item in line.strip().split(",")]
capture_start_time = datetime.strptime(metadata[metadata.index("Capture Start Time") + 1], "%Y-%m-%d %I.%M.%S.%f %p")
epoch = datetime(1, 1, 1)
ms_in_year0000 = 366 * 24 * 3600 * 1000
cst_milliseconds = (capture_start_time - epoch).total_seconds() * 1000 + ms_in_year0000 + 1.7 * 1000
data_frame = pd.read_csv(StringIO("".join(lines[6:])))
line4 = [x.strip().replace('"', '') for x in lines[3].strip().split(',')]
marker_name = 'Robot:Marker2'
first_idx = line4.index(marker_name)
marker_cols = [first_idx, first_idx + 1, first_idx + 2]
time_data = data_frame["Time (Seconds)"].to_numpy().reshape(-1, 1) * 1000 + cst_milliseconds
frame_data = data_frame["Frame"].to_numpy().reshape(-1, 1)
marker_data = data_frame.iloc[:, marker_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
time_XYZ = np.hstack((time_data, marker_data, frame_data))
time_XYZ = time_XYZ[~np.isnan(time_XYZ[:, 1]), :]

# Read from optitrack_json
with open(optitrack_json, "r") as f:
    data_optitrack = json.load(f)
qr_optitrack = []
for frame in data_optitrack:
    q = frame["q_cam"]
    r = frame["r_cam"]
    qr_optitrack.append(q + r)
qr_optitrack = np.array(qr_optitrack, dtype=float)

# Read from spad_json
with open(spad_json, "r") as f:
    data_spad = json.load(f)
qr_spad = []
for frame in data_spad:
    q = frame["q_spad_cam"]
    r = frame["r_spad_cam"]
    qr_spad.append(q + r)
qr_spad = np.array(qr_spad, dtype=float)


sat_data = np.hstack([time_XYZ, qr_optitrack, qr_spad])

# timestamp relabel and image clean
first_timestamp = float(all_images[0].replace('.png', ''))
image_timestamps = np.array([first_timestamp + 0.38 * i for i in range(len(all_images))]).reshape(-1, 1)
image_timestamps = np.delete(image_timestamps, np.arange(2565, 2600), axis=0)
image_timestamps = np.delete(image_timestamps, np.arange(2480, 2512), axis=0)
image_timestamps = np.delete(image_timestamps, np.arange(2360, 2424), axis=0)
image_timestamps = np.delete(image_timestamps, np.arange(2325, 2336), axis=0)
image_timestamps = np.delete(image_timestamps, np.arange(2245, 2249), axis=0)
all_images = np.delete(all_images, np.arange(2565, 2600), axis=0)
all_images = np.delete(all_images, np.arange(2480, 2512), axis=0)
all_images = np.delete(all_images, np.arange(2360, 2424), axis=0)
all_images = np.delete(all_images, np.arange(2325, 2336), axis=0)
all_images = np.delete(all_images, np.arange(2245, 2249), axis=0)

image_timestamps = np.delete(image_timestamps, np.arange(2510, 2548), axis=0)
image_timestamps = np.delete(image_timestamps, np.arange(2420, 2461), axis=0)
image_timestamps = np.delete(image_timestamps, np.arange(2380, 2411), axis=0)
image_timestamps = np.delete(image_timestamps, np.arange(2245, 2350), axis=0)
image_timestamps = np.delete(image_timestamps, np.arange(1986, 2005), axis=0)
all_images = np.delete(all_images, np.arange(2510, 2548), axis=0)
all_images = np.delete(all_images, np.arange(2420, 2461), axis=0)
all_images = np.delete(all_images, np.arange(2380, 2411), axis=0)
all_images = np.delete(all_images, np.arange(2245, 2350), axis=0)
all_images = np.delete(all_images, np.arange(1986, 2005), axis=0)



sat_timestamps = sat_data[:, 0].reshape(1, -1)
idx_closest = np.argmin(np.abs(image_timestamps - sat_timestamps), axis=1)
matched_data = np.hstack([image_timestamps, sat_data[idx_closest, 1:]])


# ### PnP Reprojection ###
uv_results = []
for idx, row in enumerate(matched_data):
    filename = str(image_timestamps[idx, 0])
    marker_xyz = row[1:4] * 1000       # m -> mm

    # World -> Sat
    q1 = row[5:9]                                      # xyzw
    t1 = row[9:12].reshape(3,1)                        # 3x1
    R1 = R.from_quat(q1).as_matrix()                   # 3x3
    marker_sat_space = R1.T @ (marker_xyz.reshape(3,1) - t1)  # 3x1
    axis_sat_space = np.array([[0, 0, 0], [200, 0, 0], [0, 200, 0], [0, 0, 200]]).T   # 3x4
    sat_space = np.hstack([marker_sat_space, axis_sat_space])  # 3x5

    # Sat -> Spad
    q2 = row[12:16]                    # xyzw
    t2 = row[16:19].reshape(3,1)       # 3x1, float64
    R2 = R.from_quat(q2).as_matrix()   # 3x3, float64
    spad_space = R2 @ sat_space + t2                 # 3x5
    uv, _ = cv2.projectPoints(spad_space.T, np.zeros((3,1)), np.zeros((3,1)), K, dist_coeffs)
    uv_flat = uv.reshape(-1, 2)   # return of cv2.projectPoints: (N, 1, 2), N=5; squeeze(): (N, 1, 2) -> (N, 2)
    uv_results.append([filename] + uv_flat.flatten().tolist())

uv_results = np.array(uv_results)   # N * 11


# ### Draw the axis and red dot, output ###
spad_output = r'spad_pnp_reprojection'
os.makedirs(spad_output, exist_ok=True)

for idx, row in enumerate(tqdm(uv_results, desc="Processing UV results")):
    filename = all_images[idx]
    uv_vals = row[1:].astype(float).reshape(-1, 2)
    img_path = os.path.join(spad_images_directory, os.path.basename(filename))
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skip")
        continue
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: {img_path} cannot be loaded, skip")
        continue
    u0, v0 = uv_vals[0]  # marker 1
    u1, v1 = uv_vals[1]  # origin
    u2, v2 = uv_vals[2]  # X axis
    u3, v3 = uv_vals[3]  # Y axis
    u4, v4 = uv_vals[4]  # Z axis
    # marker1, red dot
    cv2.circle(img, (int(u0), int(v0)), 3, (0, 0, 255), -1)
    # origin, red dot
    cv2.circle(img, (int(u1), int(v1)), 3, (0, 0, 255), -1)
    # XYZ axis
    cv2.line(img, (int(u1), int(v1)), (int(u2), int(v2)), (0, 0, 255), 2)   # X axis - Red
    cv2.line(img, (int(u1), int(v1)), (int(u3), int(v3)), (0, 255, 0), 2)   # Y axis - Green
    cv2.line(img, (int(u1), int(v1)), (int(u4), int(v4)), (255, 0, 0), 2)   # Z axis - Blue
    # output
    out_path = os.path.join(
        spad_output, os.path.splitext(os.path.basename(filename))[0] + "_proj.png"
    )
    cv2.imwrite(out_path, img)


# ### output aligned data
# matched_data.shape = (N, 18)
data_list = []
for row in matched_data:
    entry = {
        "Image_timestamp": float(row[0]),
        "Frame": float(row[4]),
        "Marker_XYZ": row[1:4].tolist(),
        "q_cam": row[5:9].tolist(),
        "r_cam": row[9:12].tolist(),
        "q_spad_cam": row[12:16].tolist(),
        "r_spad_cam": row[16:19].tolist()
    }
    data_list.append(entry)

with open(f"{csv_timestamp}_spad_aligned_4bit.json", "w") as f:
    json.dump(data_list, f, indent=4)
