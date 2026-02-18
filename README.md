# This project involves two types of SPAD sequence formats: 4-bit and binary files


## 4-bit sequences

### Data processing flow and py files
1. annotations_extractor.py: generate annotation with SPAD camera intrinsics K and optitrack.csv (annotation fps: 30, alige with the optitrack system)
2. xbit_visualizer.py: visualize the image
3. spad_pnp_reprojection.py: 
    i. re-label the timestamp
    ii. pnp reproection
    iii. generate the temporal aligned annotation based on re-labelled timestamp
4. Check the reprojection manually, and clean the annotation (recursive process based on reproection and visualization)

### SPAD Camera
world -> spad, obtained from cv2.solvePnP()
R_world2spad = np.array([[0.0850757, -0.02371301, 0.99609227], [0.07138514, -0.99700263, -0.02983165], [0.99381401, 0.07364413, -0.08312794]])
t_world2spad = np.array([[-189.15649423], [596.16954803], [1358.12431652]])

### Time offset when creating temporal aligned annotation:
1. ambient_full: + 1.41 * 1000
2. dark_exp1: + 0.97 * 1000
3. dark_exp2: + 2.08 * 1000
4. dark_exp3_acq00000: + 2.45 * 1000
5. dark_exp3_acq00001: + 1.23 * 1000
6. sun_N_short: + 0.97 * 1000
7. sun_SE_short_exp2: + 1.7 * 1000
8. sun_SE_short_exp3: + 3.12 * 1000

### Clean the image during to align with the annotation:
1. ambient_full: -
2. dark_exp1: -
3. dark_exp2: -
4. dark_exp3_acq00000: -
5. dark_exp3_acq00001: -
6. sun_N_short: 
    all_images = sorted([f for f in os.listdir(spad_images_directory) if f.endswith('.png')])
    all_images = np.delete(all_images, np.arange(3240, 3741), axis=0)
7. sun_SE_short_exp2: 
    all_images = sorted([f for f in os.listdir(spad_images_directory) if f.endswith('.png')])
    all_images = np.delete(all_images, np.arange(2565, 2600), axis=0)
    all_images = np.delete(all_images, np.arange(2480, 2512), axis=0)
    all_images = np.delete(all_images, np.arange(2360, 2424), axis=0)
    all_images = np.delete(all_images, np.arange(2325, 2336), axis=0)
    all_images = np.delete(all_images, np.arange(2245, 2249), axis=0)
    all_images = np.delete(all_images, np.arange(2510, 2548), axis=0)
    all_images = np.delete(all_images, np.arange(2420, 2461), axis=0)
    all_images = np.delete(all_images, np.arange(2380, 2411), axis=0)
    all_images = np.delete(all_images, np.arange(2245, 2350), axis=0)
    all_images = np.delete(all_images, np.arange(1986, 2005), axis=0)
8. sun_SE_short_exp3:     
    all_images = sorted([f for f in os.listdir(spad_images_directory) if f.endswith('.png')])
    all_images = np.delete(all_images, np.arange(4485, 4497), axis=0)
    all_images = np.delete(all_images, np.arange(4565, 4585), axis=0)
    all_images = np.delete(all_images, np.arange(4554, 4565), axis=0)

### Timestamp Relabel:
1. ambient_full:
    first_timestamp = float(all_images[0].replace('.png', ''))
    total_len = len(all_images)
    len1 = 6200
    len2 = 9500 - 6200
    len3 = 33000 - 9500
    len4 = total_len - 33000
    increments1 = np.full(len1, 0.27)
    increments2 = np.full(len2, 0.255)
    increments3 = np.full(len3, 0.27)
    increments4 = np.full(len4, 0.3)
    all_increments = np.concatenate([increments1, increments2, increments3, increments4])
    image_timestamps = first_timestamp + np.cumsum(all_increments) - all_increments[0]
    image_timestamps = image_timestamps.reshape(-1, 1)
2. dark_exp1: 
    first_timestamp = float(all_images[0].replace('.png', ''))
    image_timestamps = np.array([first_timestamp + 2.5 * i for i in range(len(all_images))]).reshape(-1, 1)
3. dark_exp2: 
    first_timestamp = float(all_images[0].replace('.png', ''))
    image_timestamps = np.array([first_timestamp + 1.5 * i for i in range(len(all_images))]).reshape(-1, 1)
4. dark_exp3_acq00000: 
    first_timestamp = float(all_images[0].replace('.png', ''))
    image_timestamps = np.array([first_timestamp + 3.5 * i for i in range(len(all_images))]).reshape(-1, 1)
5. dark_exp3_acq00001: 
    first_timestamp = float(all_images[0].replace('.png', ''))
    image_timestamps = np.array([first_timestamp + 3.5 * i for i in range(len(all_images))]).reshape(-1, 1)
6. sun_N_short: 
    first_timestamp = float(all_images[0].replace('.png', ''))
    image_timestamps = np.array([first_timestamp + 0.32 * i for i in range(len(all_images))]).reshape(-1, 1)
7. sun_SE_short_exp2: 
    first_timestamp = float(all_images[0].replace('.png', ''))
    image_timestamps = np.array([first_timestamp + 0.38 * i for i in range(len(all_images))]).reshape(-1, 1)
8. sun_SE_short_exp3: 
    first_timestamp = float(all_images[0].replace('.png', ''))
    image_timestamps = np.array([first_timestamp + 0.38 * i for i in range(len(all_images))]).reshape(-1, 1)

### The workflow for who would like to explore other bit-depth
1. Read all 4-bit images, sort by timestamp, clean the image
2. Read corresponding temporal aligned annotation
3. Then you have obtained the cleaned sequences and annotaions, with correct timestamp and temproal order
2. Extend to other bit-depth by Accumulating pixel value
3. You may need some strategy to adjust the timetamp based on your desired bit-depth
4. You may generate you own temporal aligned annotation file (.json) based on your desired bit-depth



## binary sequences

### Data processing flow and py files
1. annotations_extractor.py: generate annotation with SPAD camera intrinsics K and optitrack.csv (annotation fps: 30, alige with the optitrack system)
2. bin2png.py: clean the .bin files, convert .bin files to desired bit-depth with correct timestamp      (recursive process based on reproection and visualization)
3. spad_pnp_reprojection.py: pnp reprojection and check the correctness
4. spad_cleaned_data_generator.py: generate the temporal aligned annotation

### SPAD Camera
world -> spad, obtained from cv2.solvePnP()
1. For ambient_partial_long, ambient_partial_short, sun_SE_long
R_world2spad = np.array([[0.0959831, -0.02308601, 0.99511521], [0.12097804, -0.99204907, -0.03468372], [0.98800382, 0.12371614, -0.09242704]])
t_world2spad = np.array([[-181.92102076], [558.14488341], [1375.31565165]])
2. For sun_NW
R_world2spad = np.array([[0.0850757, -0.02371301, 0.99609227], [0.07138514, -0.99700263, -0.02983165], [0.99381401, 0.07364413, -0.08312794]])
t_world2spad = np.array([[-189.15649423], [596.16954803], [1358.12431652]])

### Time offset when creating temporal aligned annotation:
- 7200 + 2 for all sequences （-2 hours + 2 seconds）


### .bin files clean in the bin2png.py
1. ambient_partial_long:
    removed_bin = [
        1727079742.3277600,
        1727079742.4436009,
        1727079742.6628954,
        1727079742.7820358,
        1727079742.8975248,
        1727079743.7726369,
        1727079770.3243201,
        1727079780.0622597,
        1727079815.1844642,
        1727079833.1108804,
        1727079899.4342966,
        1727079901.0692236,
        1727079901.4177573,
        1727079951.3460681,
        1727079951.4932191,
        1727079951.5786583,
        1727080029.0341678,
        1727080029.5613649,
        1727080088.3672848,
        1727080110.6953282,
        1727080115.2440996,
        1727080116.2427335,
        1727080117.5190423,
        1727080165.3410215,
        1727080179.2526810,
        1727080187.6864233,
        1727080203.2243814,
        1727080218.7226524,
        1727080232.3561139,
        1727080232.4306281,
        1727080279.9898429,
        1727080280.5623608]
2. ambient_partial_short:
    removed_bin = [
        1727079527.3189344,
        1727079529.3112266,
        1727079531.9764204,
        1727079532.0830390,
        1727079542.5781138,
        1727079543.1262136]
3. sun_NW:
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
4. sun_SE_long:
    removed_bin = [
        1727081068.1331177,
        1727081068.3744123,
        1727081068.5133767,
        1727081136.9750712,
        1727081191.4518056,
        1727081191.9895220,
        1727081272.3377423,
        1727081272.4198780,
        1727081272.5669034,
        1727081272.7712300,
        1727081327.7574849,
        1727081331.6819563,
        1727081362.3859820,
        1727081362.9650683,
        1727081377.5789578,
        1727081380.5259428,
        1727081411.9561501,
        1727081412.0563772,
        1727081412.1724534,
        1727081412.2570295,
        1727081412.3885734,
        1727081412.4731860,
        1727081412.6045871,
        1727081412.7046340,
        1727081412.8202703,
        1727081412.9358008,
        1727081413.0379319,
        1727081413.6583669,
        1727081415.7225692,
        1727081415.8383095,
        1727081415.9227149,
        1727081416.0604110,
        1727081416.1477711,
        1727081416.2769380,
        1727081416.3925745,
        1727081416.4928873,
        1727081416.6280365,
        1727081416.7203178,
        1727081417.2351179,
        1727081417.2976809,
        1727081417.4089651,
        1727081417.5357101,
        1727081417.6014843,
        1727081417.7360237,
        1727081417.8362505,
        1727081417.9365966,
        1727081418.0523534,
        1727081418.1365623,
        1727081418.3148291,
        1727081418.3841233,
        1727081418.4843416,
        1727081418.6180985,
        1727081418.6983054,
        1727081418.8211739,
        1727081418.9324369,
        1727081419.0526688,
        1727081419.1442981,
        1727081419.2756059,
        1727081419.3756528,
        1727081419.4914308,
        1727081419.6071286,
        1727081419.7177603,
        1727081419.8317852,
        1727081419.9456275,
        1727081420.0445051,
        1727081420.1455021,
        1727081420.2684422,
        1727081420.3841512,
        1727081420.4843247,
        1727081420.6001732,
        1727081420.6845376,
        1727081420.8331914,
        1727081420.9205568,
        1727081421.0264027,
        1727081421.1336124,
        1727081421.2548525,
        1727081421.3549428,
        1727081421.4368086,
        1727081421.5708830,
        1727081421.6708853,
        1727081421.7937791,
        1727081421.9024334,
        1727081422.0340352,
        1727081422.1022282,
        1727081422.2406497,
        1727081422.3562500,
        1727081422.4391162,
        1727081422.5566032,
        1727081422.6724699,
        1727081422.7727628,
        1727081422.9575770,
        1727081423.0266593,
        1727081423.1371834,
        1727081423.2316031,
        1727081423.3583689,
        1727081423.5056181,
        1727081423.5586488,
        1727081423.6897879,
        1727081423.7900610,
        1727081423.9060354,
        1727081424.0000961,
        1727081424.1269336,
        1727081424.2251148,
        1727081424.3428099,
        1727081424.4553883,
        1727081424.5287652,
        1727081424.6503236,
        1727081424.7591598,
        1727081424.8627939,
        1727081424.9538050,
        1727081425.0772402,
        1727081425.1931655,
        1727081425.2932305,
        1727081425.4099419,
        1727081425.5218658,
        1727081425.6699393,
        1727081425.7328312,
        1727081425.8824723,
        1727081425.9824593,
        1727081426.0822957,
        1727081426.1979480,
        1727081426.2980208,
        1727081426.4040692,
        1727081426.5141585,
        1727081426.6141429,
        1727081426.7178137,
        1727081426.8431599,
        1727081426.9506354,
        1727081427.0563312,
        1727081427.1668415,
        1727081427.2888536,
        1727081427.3891816,
        1727081427.4892602,
        1727081427.5895360,
        1727081427.7051115,
        1727081427.8362751,
        1727081427.9444783,
        1727081428.0429003,
        1727081428.1901572,
        1727081428.2718706,
        1727081428.4099448,
        1727081428.4903054,
        1727081428.6218481,
        1727081428.7376850,
        1727081428.8066897,
        1727081428.9475007,
        1727081429.0690937,
        1727081429.1857536,
        1727081429.2736893,
        1727081429.3977129,
        1727081429.4805889,
        1727081429.5862763,
        1727081429.6983764,
        1727081429.7983665,
        1727081429.8985939,
        1727081430.0141115,
        1727081430.1296937,
        1727081430.2300193,
        1727081430.3455963,
        1727081469.4756601,
        1727081472.8849418,
        1727081536.3111658,
        1727081573.2519045]


### The workflow for who would like to explore other bit-depth
1. Clean the .bin files, generate desired bit-depth sequences with bin2png.py   
2. Generate the temporal aligned annotation with spad_cleaned_data_generator.py
3. Then you have obtained the cleaned sequences and annotaions, with correct timestamp, temproal order and desired bit-depth


## Other Information and .py files
1. The unit of 'r' in the annotation is millimeter
2. Quternion order in the annotation is xyzw
3. data_preprocessing.py: process the clean and aligned data into .pt file, split the train/val/test set. 
                          You may change the split (recommended) and modify the code based on the desired bit-depth.
4. train_GRU.py: read .pt and train on the data
5. test_GRU.py: read .pt and test the model



## .py files Intruction
1. utils/annotations_extractor.py
    The user needs to specify 
        i. SPAD camera intrinsics K (line 16 - 18)
        ii. csv_timestamp (line 22) 
    based on the corresponding sequence and bit-depth.

2. 4-bit_py_files/xbit_visualizer.py
    The user needs to specify 
        i. input_dir (line 7) 
        ii. Maximum possible pixel value (line 24)
    based on the corresponding sequence and bit-depth.

3. 4-bit_py_files/spad_pnp_reprojection.py
    The user needs to specify 
        i. spad_images_directory (line 25) 
        ii. csv_timestamp (line 30)
        iii. time offset (line 43)
        iv. timestamp relabel strategy and image clean code (line79 - 101)
    based on the corresponding sequence and bit-depth.

4. bin_py_files/bin2png.py
    The user needs to specify 
        i. desired_bitdepth (line 13)
        ii. Bin files clean code (line 25 - 51)
    based on the corresponding sequence and bit-depth.
    Note: for the bin files with wrong shape will be discard automatically

5. bin_py_files/spad_pnp_reprojection.py
    The user needs to specify 
        i. spad_images_directory (line 25) 
        ii. csv_timestamp (line 30)
        iii. time offset (line 48)
    based on the corresponding sequence and bit-depth.

6. bin_py_files/spad_cleaned_data_generator.py
    The user needs to specify 
        i. spad_images_directory (line 13) 
        ii. csv_timestamp (line 18)
        iii. time offset (line 37)
    based on the corresponding sequence and bit-depth.

7. utils/train_GRU.py
    The user needs to specify 
        i. train_set_path (line 767) 
        ii. val_set_path (line 768) 
    based on the corresponding sequence and bit-depth.

8. utils/test_GRU.py
    The user needs to specify 
        i. train_set_path (line 131) 
        ii. test_set_path (line 126) 
    based on the corresponding sequence and bit-depth.