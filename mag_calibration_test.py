from mag_calibration import *
import matplotlib.pyplot as plt


def plot_unit_sphere(axis):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    axis.plot_wireframe(x, y, z, rstride=10, cstride=10, alpha=0.5)
    axis.plot_surface(x, y, z, alpha=0.3, color='b')


def scatter_points(axis, x, y, z):
    axis.scatter(x, y, z, s=1, color='r')
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')


def show_points_data(x, y, z):
    fig = plt.figure(1)
    axis = fig.add_subplot(111, projection='3d')
    scatter_points(axis, x, y, z)
    plot_unit_sphere(axis)
    plt.show()


def calibration_reference_magnetometer_test(data):
    mag_x = data[:, 8]
    mag_y = data[:, 9]
    mag_z = data[:, 10]

    min_x = mag_x.min()
    max_x = mag_x.max()
    min_y = mag_y.min()
    max_y = mag_y.max()
    min_z = mag_z.min()
    max_z = mag_z.max()

    # q, n, d = fitEllipsoid(mag_x, mag_y, mag_z)
    # q_inv = np.linalg.inv(q)
    # b = -np.dot(q_inv, n)
    # a_inv = np.real(1 / np.sqrt(np.dot(n.T, np.dot(q_inv, n)) - d) * linalg.sqrtm(q))
    a_inv, b = calibration_reference_magnetometer(data[:, 8:])

    calib_mag_x = np.zeros(mag_x.shape)
    calib_mag_y = np.zeros(mag_y.shape)
    calib_mag_z = np.zeros(mag_z.shape)

    total_error = 0
    for i in range(len(mag_x)):
        h_hat = apply_transformation([mag_x[i], mag_y[i], mag_z[i]], a_inv, b)
        # h = np.array([[mag_x[i], mag_y[i], mag_z[i]]]).T
        # h_hat = np.matmul(a_inv, h - b)
        calib_mag_x[i] = h_hat[0]
        calib_mag_y[i] = h_hat[1]
        calib_mag_z[i] = h_hat[2]
        mag = np.dot(h_hat.T, h_hat)
        err = (mag[0][0] - 1) ** 2
        total_error += err

    print(f'minX: {min_x}, maxX: {max_x}, dX: {max_x - min_x}')
    print(f'minY: {min_y}, maxY: {max_y}, dY: {max_y - min_y}')
    print(f'minZ: {min_z}, maxZ: {max_z}, dZ: {max_z - min_z}')

    print("A_inv: ")
    print(a_inv)
    print()
    print("b")
    print(b)
    print()

    print("Total Error: %f" % total_error)

    show_points_data(mag_x, mag_y, mag_z)
    show_points_data(calib_mag_x, calib_mag_y, calib_mag_z)


def read_data():
    raw_data_0 = np.genfromtxt('./raw_calib_data/magnet-0-0-.csv', dtype=float, delimiter=',', skip_header=1)
    raw_data_1 = np.genfromtxt('./raw_calib_data/magnet-1-0-.csv', dtype=float, delimiter=',', skip_header=1)
    raw_data_2 = np.genfromtxt('./raw_calib_data/magnet-2-0-.csv', dtype=float, delimiter=',', skip_header=1)
    return raw_data_0, raw_data_1, raw_data_2


def draw_scatter(data_list):
    colors = ['r', 'b', 'g', 'y']
    markers = ['s', 'o', '^', 'd']

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    for idx, (data, color) in enumerate(data_list):
        dataX = data[:, 0]
        dataY = data[:, 1]
        dataZ = data[:, 2]
        ax1.scatter(dataX, dataY, dataZ, s=1, c=colors[idx], marker=markers[idx])

    # ax1.set_xlim3d(-100, 130)
    # ax1.set_ylim3d(-100, 130)
    # ax1.set_zlim3d(-100, 130)
    ax1.set_xlim3d(-2, 2)
    ax1.set_ylim3d(-2, 2)
    ax1.set_zlim3d(-2, 2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()


def solve_quadratic_lsm(data1, d):
    x = data1[:, 0]
    y = data1[:, 1]
    z = data1[:, 2]
    a11 = x * x
    a22 = y * y
    a33 = z * z
    a12 = 2 * x * y
    a23 = 2 * y * z
    a13 = 2 * x * z
    a14 = 2 * x
    a24 = 2 * y
    a34 = 2 * z
    a44 = np.ones(x.shape)
    H1 = np.array([a11, a22, a33, a12, a23, a13, a14, a24, a34, a44])
    H2 = np.matmul(H1, H1.T)
    H2inv = np.linalg.inv(H2)
    H3 = np.matmul(H1.T, H2inv)
    res = np.matmul(d, H3)
    print(res)
    return res


def quadratic(x, y, z, t):
    a11 = t[0]
    a22 = t[1]
    a33 = t[2]
    a12 = t[3]
    a23 = t[4]
    a13 = t[5]
    a14 = t[6]
    a24 = t[7]
    a34 = t[8]
    a44 = t[9]
    res = (a11 * x * x) + (a22 * y * y) + (a33 * z * z) + (2 * a12 * x * y) \
          + (2 * a23 * y * z) + (2 * a13 * x * z) + (2 * a14 * x) \
          + (2 * a24 * y) + (2 * a34 * z) + a44
    return res


def transform_data_quadratic(src, tx, ty, tz):
    res = np.zeros(src.shape)
    for i in range(src.shape[0]):
        res[i, 0] = quadratic(src[i, 0], src[i, 1], src[i, 2], tx)
        res[i, 1] = quadratic(src[i, 0], src[i, 1], src[i, 2], ty)
        res[i, 2] = quadratic(src[i, 0], src[i, 1], src[i, 2], tz)
    return res


def main_visualise():
    data0, data1, data2 = read_data()
    d0 = data0[:, 8:]
    d1 = data1[:, 8:]

    a_inv, b = calibration_reference_magnetometer(d0)
    data_ref = apply_transformation_to_dataset(d0, a_inv, b)

    transform_array1 = get_linear_transformation(d1, data_ref)
    d1_lin = transform_data_linear(d1, transform_array1)

    tx1 = solve_quadratic_lsm(d1, data_ref[:, 0])
    ty1 = solve_quadratic_lsm(d1, data_ref[:, 1])
    tz1 = solve_quadratic_lsm(d1, data_ref[:, 2])
    d1_quad = transform_data_quadratic(d1, tx1, ty1, tz1)

    # data = [(d0, 'r'), (d1, 'g'), (d2, 'b')]
    data = [(data_ref, 'r'), (d1_quad, 'g'), (d1_lin, 'b')]
    draw_scatter(data)


def main():
    # imu,time,ax,ay,az,gx,gy,gz,mx,my,mz
    raw_data = np.genfromtxt('./raw_calib_data/magnet-0-0-.csv', dtype=float, delimiter=',', skip_header=1)
    calibration_reference_magnetometer_test(raw_data)


if __name__ == '__main__':
    main_visualise()
    # main()
