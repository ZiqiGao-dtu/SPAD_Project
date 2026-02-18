import numpy as np
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import cv2
import pandas as pd
from io import StringIO
import json


# world -> orbbec from cv2.solvePnP()
R_world2orbbec = np.array([[0.20316333, -0.03700384, 0.97844539], [-0.01141671, -0.99930722, -0.03542226], [0.9790783, -0.00397412, -0.20344505]])
t_world2orbbec = np.array([[-289.80138202], [426.05225515], [1440.07510639]])


# world -> spad from cv2.solvePnP()
R_world2spad = np.array([[0.0850757, -0.02371301, 0.99609227], [0.07138514, -0.99700263, -0.02983165], [0.99381401, 0.07364413, -0.08312794]])
t_world2spad = np.array([[-189.15649423], [596.16954803], [1358.12431652]])


# sat -> world
csv_timestamp = "Take 2024-09-20 11.23.20 AM"
optitrack_csv = f"{csv_timestamp}.csv"
with open(optitrack_csv, "r") as f:
    lines = f.readlines()
data_frame = pd.read_csv(StringIO("".join(lines[6:])))
line4 = [x.strip().replace('"', '') for x in lines[3].strip().split(',')]
marker_name = 'Robot'
first_idx = line4.index(marker_name)
marker_cols = [first_idx, first_idx + 1, first_idx + 2, first_idx + 3, first_idx + 4, first_idx + 5, first_idx + 6]
marker_data = data_frame.iloc[:, marker_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
frame_data = data_frame["Frame"].to_numpy().reshape(-1, 1)
time_data = data_frame["Time (Seconds)"].to_numpy().reshape(-1, 1)
marker_data = np.hstack((frame_data, time_data, marker_data))
marker_data = marker_data[~np.isnan(marker_data[:, 3]), :]
q_sat2world = marker_data[:, 2:6]
R_sat2world = np.array([R.from_quat(q).as_matrix() for q in q_sat2world])
t_sat2world = marker_data[:, 6:9] * 1000.0


# we would like: sat->world->orbbec
N = R_sat2world.shape[0]
R_sat2orbbec = np.array([R_world2orbbec @ R_sat2world[i] for i in range(N)])  # (N,3,3)
t_sat2orbbec = np.array([[R_world2orbbec @ t_sat2world[i].reshape(3, 1) + t_world2orbbec for i in range(N)]]).reshape(-1,3)  # (N,3,1) -> (N, 3)

# we would like: sat->world->spad
R_sat2spad = np.array([R_world2spad @ R_sat2world[i] for i in range(N)])  # (N,3,3)
t_sat2spad = np.array([[R_world2spad @ t_sat2world[i].reshape(3, 1) + t_world2spad for i in range(N)]]).reshape(-1,3)  # (N,3,1) -> (N, 3)


# output orbbec json
json_list1 = []
for i in range(N):
    frame = int(marker_data[i,0])
    time_sec = float(marker_data[i,1])
    q = R.from_matrix(R_sat2orbbec[i]).as_quat()
    r = t_sat2orbbec[i].tolist()
    json_entry = {
        "Frame": str(frame),
        "Time(seconds)": str(time_sec),
        "q_orbbec_cam": q.tolist(),
        "r_orbbec_cam": r
    }
    json_list1.append(json_entry)
output_file1 = f"{csv_timestamp}_orbbec.json"
with open(output_file1, "w") as f:
    json.dump(json_list1, f, indent=2)
print(f"orbbec.json generated: {output_file1}")


# output spad json
json_list2 = []
for i in range(N):
    frame = int(marker_data[i,0])
    time_sec = float(marker_data[i,1])
    q = R.from_matrix(R_sat2spad[i]).as_quat()
    r = t_sat2spad[i].tolist()
    json_entry = {
        "Frame": str(frame),
        "Time(seconds)": str(time_sec),
        "q_spad_cam": q.tolist(),
        "r_spad_cam": r
    }
    json_list2.append(json_entry)
output_file2 = f"{csv_timestamp}_spad.json"
with open(output_file2, "w") as f:
    json.dump(json_list2, f, indent=2)
print(f"spad.json generated: {output_file2}")


# output optitrack json
json_list3 = []
for i in range(N):
    frame = int(marker_data[i, 0])
    time_sec = float(marker_data[i, 1])
    q = marker_data[i, 2:6]
    r = (marker_data[i, 6:9] * 1000.0).tolist()
    json_entry = {
        "Frame": str(frame),
        "Time(seconds)": str(time_sec),
        "q_cam": q.tolist(),
        "r_cam": r
    }
    json_list3.append(json_entry)
output_file3 = f"{csv_timestamp}_optitrack.json"
with open(output_file3, "w") as f:
    json.dump(json_list3, f, indent=2)
print(f"optitrack.json generated: {output_file3}")
