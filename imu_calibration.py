import pandas as pd
from os import walk
from arduino_common import *
from mag_calibration import *

data_dir = './raw_calib_data/'


def get_available_sensors():
    imus = []
    all_file_names = next(walk(data_dir), (None, None, []))[2]
    for file_name in all_file_names:
        parts = file_name.split('-')
        if len(parts) != 4:
            continue
        file_type = parts[0]
        imu = parts[1]
        if file_type == "gyro_accel":
            if imu not in imus:
                imus.append(imu)
    return imus


def get_files(_file_type, _imu):
    file_names = []
    all_file_names = next(walk(data_dir), (None, None, []))[2]
    for file_name in all_file_names:
        parts = file_name.split('-')
        if len(parts) != 4:
            continue
        file_type = parts[0]
        imu = parts[1]
        side = parts[2]
        if file_type == _file_type and imu == _imu:
            file_names.append(file_name)
    return file_names


def calibrate_gyroscope(file_names, gyro_range):
    scale = get_gyro_scale(gyro_range)
    counter = 0
    total_x = 0
    total_y = 0
    total_z = 0
    for file_name in file_names:
        df = pd.read_csv(f"{data_dir}/{file_name}", dtype=int)
        for index, row in df.iterrows():
            total_x += row['gx'] * scale
            total_y += row['gy'] * scale
            total_z += row['gz'] * scale
            counter += 1

    bias_x = total_x / counter
    bias_y = total_y / counter
    bias_z = total_z / counter

    params = {'bias': {'x': bias_x, 'y': bias_y, 'z': bias_z},
              'sensitivity': {'x': 1, 'y': 1, 'z': 1},
              'min': {'x': 0, 'y': 0, 'z': 0},
              'max': {'x': 0, 'y': 0, 'z': 0}}

    return params


def calibrate_accelerometer(file_names, accel_range):
    scale = get_accel_scale(accel_range)
    counter = 0
    total_x = 0
    total_y = 0
    total_z = 0
    for file_name in file_names:
        df = pd.read_csv(f"{data_dir}/{file_name}", dtype=int)
        for index, row in df.iterrows():
            total_x += row['ax'] * scale
            total_y += row['ay'] * scale
            total_z += row['az'] * scale
            counter += 1

    bias_x = total_x / counter
    bias_y = total_y / counter
    bias_z = total_z / counter

    # X
    max_x = 0
    min_x = 0
    if bias_x > max_x:
        max_x = bias_x
    if bias_x < min_x:
        min_x = bias_x

    bias_x = (min_x + max_x) / 2.0
    sensitivity_x = gravity / ((abs(min_x) + abs(max_x)) / 2.0)

    # Y
    max_y = 0
    min_y = 0
    if bias_y > max_y:
        max_y = bias_y
    if bias_y < min_y:
        min_y = bias_y

    bias_y = (min_y + max_y) / 2.0
    sensitivity_y = gravity / ((abs(min_y) + abs(max_y)) / 2.0)

    # Z
    max_z = 0
    min_z = 0
    if bias_z > max_z:
        max_z = bias_z
    if bias_z < min_z:
        min_z = bias_z

    bias_z = (min_z + max_z) / 2.0
    sensitivity_z = gravity / ((abs(min_z) + abs(max_z)) / 2.0)

    params = {'bias': {'x': bias_x, 'y': bias_y, 'z': bias_z},
              'sensitivity': {'x': sensitivity_x, 'y': sensitivity_y, 'z': sensitivity_z},
              'min': {'x': min_x, 'y': min_y, 'z': min_z},
              'max': {'x': max_x, 'y': max_y, 'z': max_z}}
    return params


def calibrate_reference_magnetometer(data):
    a_inv, b = calibration_reference_magnetometer(data)
    return a_inv, b


def calibrate_magnetometer_by_reference(data, data_ref):
    # transform:
    # [a11, a12, a13, b1]
    # [a21, a22, a33, b2]
    # [a31, a32, a33, b3]
    transform = get_linear_transformation(data, data_ref)
    a = transform[:, :3]
    b = transform[:, 3]
    return a, b
