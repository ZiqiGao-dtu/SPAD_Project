import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import time
from PIL import Image
import json
import os
import numpy as np
from glob import glob
import sys
import random
from tqdm import tqdm


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# SPAD 4Bit Sequences, 12 Sequences (dark 3 with 2 Sequences)
spad_4bit_train = [
    (r"ambient_runs\ambient_partial_long\spad\4bit",
     r"ambient_runs\ambient_partial_long\Take 2024-09-23 10.22.01 AM_spad_aligned_cleaned_4bit.json"),
    (r"ambient_runs\ambient_partial_short\spad\4bit",
     r"ambient_runs\ambient_partial_short\Take 2024-09-23 10.18.22 AM_spad_aligned_cleaned_4bit.json"),
    (r"sun_NW\spad\4bit",
     r"sun_NW\Take 2024-09-23 11.49.58 AM_spad_aligned_cleaned_4bit.json"),
    (r"sun_SE_long\spad\4bit",
     r"sun_SE_long\Take 2024-09-23 10.43.57 AM_spad_aligned_cleaned_4bit.json"),
    (r"dark_runs\dark_exp3\spad\4bit\acq00000",
     r"dark_runs\dark_exp3\Take 2024-09-20 12.12.19 PM_spad_aligned_cleaned_4bit_00000.json"),
    (r"dark_runs\dark_exp3\spad\4bit\acq00001",
     r"dark_runs\dark_exp3\Take 2024-09-20 12.12.19 PM_spad_aligned_cleaned_4bit_00001.json"),
    (r"dark_runs\dark_exp2\spad\4bit",
     r"dark_runs\dark_exp2\Take 2024-09-20 12.10.35 PM_spad_aligned_cleaned_4bit.json"),
    (r"sun_N_short\spad\4bit",
     r"sun_N_short\Take 2024-09-20 01.11.24 PM_spad_aligned_cleaned_4bit.json"),
    (r"sun_SE_short_exp3\spad\4bit",
     r"sun_SE_short_exp3\Take 2024-09-20 11.56.52 AM_spad_aligned_cleaned_4bit.json"),
    (r"ambient_runs\ambient_full\spad\4bit",
     r"ambient_runs\ambient_full\Take 2024-09-20 11.23.20 AM_spad_aligned_cleaned_4bit.json"),
    (r"ambient_runs\ambient_partial_long\spad\2bit_sf15",
     r"ambient_runs\ambient_partial_long\Take 2024-09-23 10.22.01 AM_spad_aligned_cleaned_2bit_sf15.json"),
    (r"ambient_runs\ambient_partial_short\spad\2bit_sf15",
     r"ambient_runs\ambient_partial_short\Take 2024-09-23 10.18.22 AM_spad_aligned_cleaned_2bit_sf15.json"),
    (r"sun_NW\spad\2bit_sf15",
     r"sun_NW\Take 2024-09-23 11.49.58 AM_spad_aligned_cleaned_2bit_sf15.json"),
    (r"sun_SE_long\spad\2bit_sf15",
     r"sun_SE_long\Take 2024-09-23 10.43.57 AM_spad_aligned_cleaned_2bit_sf15.json"),
]

spad_4bit_test = [
    (r"dark_runs\dark_exp1\spad\4bit",
     r"dark_runs\dark_exp1\Take 2024-09-20 12.09.01 PM_spad_aligned_cleaned_4bit.json"),
    (r"ambient_runs\ambient_full\spad\4bit",
     r"ambient_runs\ambient_full\Take 2024-09-20 11.23.20 AM_spad_aligned_cleaned_4bit.json"),
    (r"sun_SE_short_exp2\spad\4bit",
     r"sun_SE_short_exp2\Take 2024-09-20 11.50.48 AM_spad_aligned_cleaned_4bit.json")
]


def load_and_build_dataset(train_sequences, test_sequences, image_type, train_window_length=128, stride=8):
    """
    train_list: list of sliding windows, each window is a dict containing
    val_list and test_list: lists of dicts, each dict represents a full sequence
    """
    # *** Train set *** #
    train_list = []
    train_r_raw = []
    train_delta_t_raw = []
    # ===== 1. read JSON and sort it =====
    print(f"\n[INFO] Loading training sequences for {image_type} ...")
    for seq_dir, json_path in tqdm(train_sequences):
        with open(json_path, "r") as f:
            data = json.load(f)
        data_sorted = sorted(data, key=lambda x:x['Image_timestamp'])
        # ===== 2. build img_path =====
        img_paths = sorted(glob(os.path.join(seq_dir, "*.png")))
        # data cleaning
        if "sun_N_short" in seq_dir:
            del img_paths[3240:3741]
        elif "sun_SE_short_exp3" in seq_dir:
            del img_paths[4485:4497]
            del img_paths[4554:4585]
        if len(img_paths) != len(data_sorted):
            raise ValueError(f"Image count mismatch: {seq_dir}")
        N = len(img_paths)
        if "ambient_full" in seq_dir:
            data_sorted = data_sorted[:N//2]
            img_paths = img_paths[:N//2]
            N = N//2
        # ===== 3. Save training frame r for calculating mean/std =====
        timestamps = [float(item['Image_timestamp']) for item in data_sorted]
        delta_ts_all = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        train_delta_t_raw.extend(delta_ts_all)
        for item in data_sorted:
            r = item['r_spad_cam']
            train_r_raw.append(torch.tensor(r, dtype=torch.float32))
        # ===== 5. Build train set based on windows =====
        for i in range(1, N - train_window_length + 1, stride):
            window_paths = img_paths[i:i + train_window_length]
            window_states = []
            for j, item in enumerate(data_sorted[i:i + train_window_length]):
                q_prev = data_sorted[i + j - 1]['q_spad_cam']
                r_prev = data_sorted[i + j - 1]['r_spad_cam']
                delta_t = float(item['Image_timestamp']) - float(data_sorted[i + j - 1]['Image_timestamp'])
                window_states.append(torch.tensor(q_prev + r_prev + [delta_t], dtype=torch.float32))

            target_item = data_sorted[i + train_window_length - 1]
            target_state = torch.tensor(target_item['q_spad_cam'] + target_item['r_spad_cam'], dtype=torch.float32)
            train_list.append({
                "img_paths": window_paths,
                "state_seq": torch.stack(window_states, dim=0),
                "target_state": target_state
            })
    # ===== 6. Compute train r mean/std and delta_t mean/std=====
    all_r_tensor = torch.stack(train_r_raw, dim=0)  # shape: [num_frames, 3]
    r_mean = all_r_tensor.mean(dim=0)
    r_std = all_r_tensor.std(dim=0)
    delta_t_tensor = torch.tensor(train_delta_t_raw, dtype=torch.float32)
    delta_t_mean = delta_t_tensor.mean()
    delta_t_std = delta_t_tensor.std()

    # ===== 7. Standardize r in train =====
    def standardize_train(item):
        if 'state_seq' in item:
            # Standardize all frames in the window: r (4:7) and delta_t (7)
            item['state_seq'][:, 4:7] = (item['state_seq'][:, 4:7] - r_mean) / r_std
            item['state_seq'][:, 7] = (item['state_seq'][:, 7] - delta_t_mean) / delta_t_std
        if 'target_state' in item:
            item['target_state'][4:7] = (item['target_state'][4:7] - r_mean) / r_std
        return item
    train_list = [standardize_train(item) for item in train_list]
    random.shuffle(train_list)

    # *** Validation set / Testing set *** #
    val_list, test_list = [], []
    # ===== 1. read JSON and sort it =====
    print(f"\n[INFO] Loading test sequences for {image_type} ...")
    for seq_dir, json_path in tqdm(test_sequences):
        with open(json_path, "r") as f:
            data = json.load(f)
        data_sorted = sorted(data, key=lambda x: x['Image_timestamp'])
        # ===== 2. build img_path =====
        img_paths = sorted(glob(os.path.join(seq_dir, "*.png")))
        if "sun_SE_short_exp2" in seq_dir:
            del img_paths[2565:2600]
            del img_paths[2480:2512]
            del img_paths[2360:2424]
            del img_paths[2325:2336]
            del img_paths[2245:2249]
            del img_paths[2510:2548]
            del img_paths[2420:2461]
            del img_paths[2380:2411]
            del img_paths[2245:2350]
            del img_paths[1986:2005]
        if len(img_paths) != len(data_sorted):
            raise ValueError(f"Image count mismatch: {seq_dir}")
        N = len(img_paths)

        # ===== 3. Build val/test set =====
        def build_full_sequence(start_idx, end_idx):
            seq_img_paths = img_paths[start_idx+1:end_idx]
            seq_timestamps = [float(item['Image_timestamp']) for item in data_sorted[start_idx:end_idx]]
            init_qr = torch.tensor(data_sorted[start_idx]['q_spad_cam'] + data_sorted[start_idx]['r_spad_cam'], dtype=torch.float32)
            seq_delta_ts = [seq_timestamps[i] - seq_timestamps[i-1] for i in range(1, len(seq_timestamps))]
            seq_targets = [torch.tensor(item['q_spad_cam'] + item['r_spad_cam'], dtype=torch.float32)
                           for item in data_sorted[start_idx+1:end_idx]]
            return {
                "img_paths": seq_img_paths,
                "delta_ts": seq_delta_ts,
                "target_states": torch.stack(seq_targets, dim=0),
                "init_qr": init_qr
            }
        if "ambient_full" in seq_dir:
            start_idx = N // 2
            mid_idx = (N + start_idx) // 2
            val_list.append(build_full_sequence(start_idx, mid_idx))
            test_list.append(build_full_sequence(mid_idx, N))
        else:
            mid_idx = N // 2  # split in half
            val_list.append(build_full_sequence(0, mid_idx))
            test_list.append(build_full_sequence(mid_idx, N))

    # ===== 4. Standardize r in train/val/test =====
    def standardize_test(item):
        if 'target_states' in item:
            item['target_states'][:, 4:7] = (item['target_states'][:, 4:7] - r_mean) / r_std
        # Standardize delta_t of val/test
        if 'delta_ts' in item:
            delta_t_arr = torch.tensor(item['delta_ts'], dtype=torch.float32)
            delta_t_arr = (delta_t_arr - delta_t_mean) / delta_t_std
            item['delta_ts'] = delta_t_arr.tolist()
        if 'init_qr' in item:
            item['init_qr'][4:7] = (item['init_qr'][4:7] - r_mean) / r_std

        return item
    val_list = [standardize_test(item) for item in val_list]
    test_list = [standardize_test(item) for item in test_list]

    return train_list, val_list, test_list, r_mean, r_std, delta_t_mean, delta_t_std


# save data to local folder
def save_dataset(train_list, val_list, test_list, r_mean, r_std, dataset_type, dt_mean, dt_std):
    os.makedirs("processed_data", exist_ok=True)
    train_data_to_save = {
        "data": train_list,
        "r_mean": r_mean,
        "r_std": r_std,
        "delta_t_mean": dt_mean,
        "delta_t_std": dt_std
    }
    torch.save(train_data_to_save, f"processed_data/{dataset_type}_train.pt")
    torch.save(val_list, f"processed_data/{dataset_type}_val.pt")
    torch.save(test_list, f"processed_data/{dataset_type}_test.pt")
    print(f"Saved {dataset_type} dataset, train:{len(train_list)}, val:{len(val_list)}, test:{len(test_list)}")


# Main
def main(dataset_type):

    train_window_length = 32
    stride = 4

    # ===== Map dataset_type to sequences =====
    train_sequences, test_sequences = spad_4bit_train, spad_4bit_test

    # ===== Load and build dataset =====
    train_list, val_list, test_list, r_mean, r_std, dt_mean, dt_std = load_and_build_dataset(
        train_sequences, test_sequences, dataset_type, train_window_length=train_window_length, stride=stride
    )

    # ===== save to folder =====
    save_dataset(train_list, val_list, test_list, r_mean, r_std, dataset_type, dt_mean, dt_std)


if __name__ == "__main__":
    for dtype in ["spad4bit"]:#["spad1bit", "spad2bit", "spad4bit"]:
        main(dtype)
