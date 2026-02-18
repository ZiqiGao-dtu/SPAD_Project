import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime


# ### Setting ###
bin_path = r'spad\raw'
bin_files = [f for f in os.listdir(bin_path) if f.endswith(".bin")]
bin_files.sort()  # ensure time order
desired_bitdepth = 4
output_path = os.path.join(os.path.dirname(bin_path), f"{desired_bitdepth}bit_sf")
if not os.path.exists(output_path):
    os.makedirs(output_path)
frames_per_bin = 256
rows, cols = 512, 512
exp_time = 4e-3  # milliseconds
# 3 * 1bit = 2bit
# 15 * 1bit = 4bit
# 255 * 1bit = 8bit
# 17 * 4bit = 8bit
frames_per_img = 2 ** desired_bitdepth - 1
removed_bin = [
    1727085080.6439650,
    1727085081.7546954,
    1727085082.4243555,
    1727085082.9722407,
    1727085141.0852613,
    1727085155.1897538,
    1727085230.7654753,
    1727085283.3797975,
    1727085304.9102986,
    1727085321.7639472,
    1727085354.3490281,
    1727085369.7303882,
    1727085396.7153754,
    1727085397.2598650,
    1727085411.8245220,
    1727085420.0681176,
    1727085431.9725602,
    1727085443.8847442,
    1727085466.0545411,
    1727085485.7018039,
    1727085517.1708324,
    1727085523.5671005,
    1727085531.7209303,
    1727085571.5168025,
    1727085571.5901687,
    1727085599.2107043]


# ### ========= load bin file ========= ###
def read_512Sbin(bin_file, rows=512, cols=512, num_frames=256):
    # bin_file shape = (num_frames, rows, cols)
    data = np.fromfile(bin_file, dtype=np.uint8)  # e.g. (2031620,) byte
    total_bits = rows * cols * num_frames         # e.g. (16252960,) bit
    bits = np.unpackbits(data)[:total_bits]
    arr = bits.reshape((num_frames, rows, cols))
    arr = np.transpose(arr, (0, 2, 1))
    arr = np.flip(arr, axis=1)
    return arr


# ### ========= main loop ========= ###
if desired_bitdepth in [1, 2, 4]:
    file_miss_matched = []
    file_shape_error = []
    for _, bin_file in enumerate(tqdm(bin_files, desc="Processing bin files")):
        full_bin_path = os.path.join(bin_path, bin_file)
        # --- Load timestamp from filename ---
        # format: RAW_<sec>.bin, unit: second
        match = re.match(r"^RAW_(\d+)\.(\d+)\.bin$", bin_file)
        if match:
            sec = match.group(1)   # e.g. "1694852387"
            frac = match.group(2)  # e.g. "123"
            base_timestamp_s = float(f"{sec}.{frac}")
            if base_timestamp_s in removed_bin:
                continue

            epoch_0001 = datetime(1, 1, 1, 0, 0, 0)
            epoch_unix = datetime(1970, 1, 1, 0, 0, 0)
            delta_ms = (epoch_unix - epoch_0001).total_seconds() * 1000
            ms_in_year0000 = 366 * 24 * 3600 * 1000

            base_timestamp_ms = base_timestamp_s * 1000 + delta_ms + ms_in_year0000
        else:
            file_miss_matched.append(bin_file)
            continue

        # --- load bin data ---
        try:
            frames = read_512Sbin(full_bin_path, rows, cols, frames_per_bin)
        except ValueError as e:  # if the bin file has a wrong shape - ValueError
            file_shape_error.append(bin_file)
            continue

        # --- divide bin file into n-bit image ---
        num_imgs = frames_per_bin // frames_per_img  # discard last if not divisible

        for i in range(num_imgs):
            start_idx = i * frames_per_img
            end_idx = (i + 1) * frames_per_img
            subarray = frames[start_idx:end_idx, :, :]

            # --- Accumulate to get n-bit image ---
            nbit_img = np.sum(subarray.astype(np.uint16), axis=0)
            img = Image.fromarray(nbit_img.astype(np.uint8), mode="L")

            # --- output file ---
            img_timestamp = base_timestamp_ms + i * frames_per_img * exp_time
            out_name = f"{img_timestamp:.7f}.png"
            img.save(os.path.join(output_path, out_name))
    print("Miss matched files: " + str(file_miss_matched))
    print("File with shape error: " + str(file_shape_error))
else:
    pass




