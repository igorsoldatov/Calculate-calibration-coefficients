from imu_calibration import *
import json


def main(name):
    accel_range = AccelRange.ACCEL_RANGE_16G
    gyro_range = GyroRange.GYRO_RANGE_2000DPS

    calib_params = {}

    imus = get_available_sensors()
    for imu in imus:
        gyro_file_names = get_files("gyro_accel", imu)
        gyro_calibration_coefficients = calibrate_gyroscope(gyro_file_names, gyro_range)
        accel_calibration_coefficients = calibrate_accelerometer(gyro_file_names, accel_range)
        print(f"IMU: {imu}")
        print("Gyroscope calibration coefficients:")
        print(gyro_calibration_coefficients)
        print("Accelerometer calibration coefficients:")
        print(accel_calibration_coefficients)
        print("\n")
        calib_params[imu] = {"IMU": imu,
                             "reference": False,
                             "gyroscope": gyro_calibration_coefficients,
                             "accelerometer": accel_calibration_coefficients,
                             "magnetometer": {}}

    ref_imu = '0'
    gyro_file_names = get_files("magnet", ref_imu)
    data = np.genfromtxt(f'./raw_calib_data/{gyro_file_names[0]}', dtype=float, delimiter=',', skip_header=1)
    scaled_data = scaling_magnetometer_data(data[:, 8:])
    a_inv_ref, b_ref = calibrate_reference_magnetometer(scaled_data)
    ref_data = apply_transformation_to_dataset(scaled_data, a_inv_ref, b_ref)
    magnet_ref_calibration_coefficients = {"A": a_inv_ref.tolist(), "b": b_ref.tolist()}
    print(f"Ref IMU: {ref_imu}")
    print("Reference magnetometer calibration coefficients:")
    print(magnet_ref_calibration_coefficients)
    print("\n")
    calib_params[ref_imu]["reference"] = True
    calib_params[ref_imu]["magnetometer"] = magnet_ref_calibration_coefficients

    for imu in imus:
        if imu not in ref_imu:
            gyro_file_names = get_files("magnet", imu)
            data = np.genfromtxt(f'./raw_calib_data/{gyro_file_names[0]}', dtype=float, delimiter=',',
                                 skip_header=1)
            scaled_imu_data = scaling_magnetometer_data(data[:, 8:])
            a, b = calibrate_magnetometer_by_reference(scaled_imu_data, ref_data)
            magnet_calibration_coefficients = {"A": a.tolist(), "b": b.tolist()}
            print(f"IMU: {imu}")
            print("Magnetometer calibration coefficients:")
            print(magnet_calibration_coefficients)
            print("\n")
            calib_params[imu]["magnetometer"] = magnet_calibration_coefficients

    print("All calibration coefficients:")
    print(calib_params)

    with open("imu_calibration_coefficients.json", "w") as outfile:
        json.dump(calib_params, outfile)


if __name__ == '__main__':
    main('PyCharm')
