import numpy as np
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import cv2
import pandas as pd
from io import StringIO
import json
from tqdm import tqdm


# ### Load all image names ###
spad_images_directory = r'spad\4bit'
all_images = sorted([f for f in os.listdir(spad_images_directory) if f.endswith('.png')])  # Sorted list of JPG file names


# ### Read data from .csv and .json ###
csv_timestamp = "Take 2024-09-23 11.49.58 AM"
optitrack_csv = f"{csv_timestamp}.csv"
optitrack_json = f"annotations/{csv_timestamp}_optitrack.json"
spad_json = f"annotations/{csv_timestamp}_spad.json"


# Read from optitrack_csv
with open(optitrack_csv, "r") as f:
    lines = f.readlines()
metadata = [item for line in lines[:6] for item in line.strip().split(",")]
capture_start_time = datetime.strptime(metadata[metadata.index("Capture Start Time") + 1], "%Y-%m-%d %I.%M.%S.%f %p")
epoch = datetime(1, 1, 1)
ms_in_year0000 = 366 * 24 * 3600 * 1000
cst_milliseconds = (capture_start_time - epoch).total_seconds() * 1000 + ms_in_year0000
data_frame = pd.read_csv(StringIO("".join(lines[6:])))
line4 = [x.strip().replace('"', '') for x in lines[3].strip().split(',')]
marker_name = 'Robot:Marker2'
first_idx = line4.index(marker_name)
marker_cols = [first_idx, first_idx + 1, first_idx + 2]
time_data = data_frame["Time (Seconds)"].to_numpy().reshape(-1, 1) * 1000 + cst_milliseconds - 7200*1000 + 2*1000
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


# ### Integrate images with satellite data based on timestamps ###
# ### Select the satellite data entry closest to each image timestamp for PnP reprojection ###
image_timestamps = np.array([round(float(fname.replace('.png','')), 7) for fname in all_images]).reshape(-1,1)
sat_timestamps = sat_data[:, 0].reshape(1, -1)  # 1 x m

chunk_size = 5000
desired_bitdepth = os.path.basename(spad_images_directory)
output_file = f"{csv_timestamp}_spad_aligned_cleaned_{desired_bitdepth}.json"

with open(output_file, "w") as f:
    f.write("[\n")

    first_entry = True
    for start in tqdm(range(0, len(all_images), chunk_size), desc="Processing images"):
        end = min(start + chunk_size, len(all_images))
        image_timestamps = np.array([float(fname.replace('.png', '')) for fname in all_images[start:end]]).reshape(-1, 1)
        idx_closest = np.argmin(np.abs(image_timestamps - sat_timestamps), axis=1)
        matched_data = np.hstack([image_timestamps, sat_data[idx_closest, 1:]])

        for row in matched_data:
            entry = {
                "Image_timestamp": f"{row[0]:.7f}",
                "Frame": float(row[4]),
                "Marker_XYZ": row[1:4].tolist(),
                "q_cam": row[5:9].tolist(),
                "r_cam": row[9:12].tolist(),
                "q_spad_cam": row[12:16].tolist(),
                "r_spad_cam": row[16:19].tolist()
            }
            if not first_entry:
                f.write(",\n")
            json.dump(entry, f, indent=4)
            first_entry = False

    f.write("\n]\n")


