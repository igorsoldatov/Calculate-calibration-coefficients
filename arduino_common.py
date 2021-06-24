from enum import Enum


gravity = 9.807


class AccelRange(Enum):
   ACCEL_RANGE_2G = 1
   ACCEL_RANGE_4G = 2
   ACCEL_RANGE_8G = 3
   ACCEL_RANGE_16G = 4


class GyroRange(Enum):
   GYRO_RANGE_250DPS = 1
   GYRO_RANGE_500DPS = 2
   GYRO_RANGE_1000DPS = 3
   GYRO_RANGE_2000DPS = 4


def get_gyro_scale(gyro_range):
    scale = 0.0
    if gyro_range == GyroRange.GYRO_RANGE_250DPS:
        scale = 131.0 / 250.0
    elif gyro_range == GyroRange.GYRO_RANGE_500DPS:
        scale = 65.5 / 500.0
    elif gyro_range == GyroRange.GYRO_RANGE_1000DPS:
        scale = 32.8 / 1000.0
    elif gyro_range == GyroRange.GYRO_RANGE_2000DPS:
        scale = 16.4 / 2000.0
    return scale


def get_accel_scale(accel_range):
    range_value = 0.0
    if accel_range == AccelRange.ACCEL_RANGE_2G:
        range_value = 2.0
    elif accel_range == AccelRange.ACCEL_RANGE_4G:
        range_value = 4.0
    elif accel_range == AccelRange.ACCEL_RANGE_8G:
        range_value = 8.0
    elif accel_range == AccelRange.ACCEL_RANGE_16G:
        range_value = 16.0
    scale = gravity * range_value / 32767.5
    return scale
