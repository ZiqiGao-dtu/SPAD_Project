import os
import numpy as np
from PIL import Image
from tqdm import tqdm


'''
Take the 4bit as the example:
'''

input_dir = r"spad\4bit"
output_dir = r"scaled4bit"
os.makedirs(output_dir, exist_ok=True)


for filename in tqdm(os.listdir(input_dir), desc="Processing images"):

    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)
    img_np = np.array(img, dtype=np.uint8)

    # Normalize to 0–1
    # Scale to 0–255
    img_vis = (img_np / 15 * 255).astype(np.uint8)


    out_path = os.path.join(output_dir, filename)
    Image.fromarray(img_vis).save(out_path)

