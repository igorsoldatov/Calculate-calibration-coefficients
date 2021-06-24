import numpy as np
from scipy import linalg


def fit_ellipsoid(x, y, z):
    a1 = x ** 2
    a2 = y ** 2
    a3 = z ** 2
    a4 = 2 * np.multiply(y, z)
    a5 = 2 * np.multiply(x, z)
    a6 = 2 * np.multiply(x, y)
    a7 = 2 * x
    a8 = 2 * y
    a9 = 2 * z
    a10 = np.ones(len(x)).T
    D = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])

    # Eqn 7, k = 4
    C1 = np.array([[-1, 1, 1, 0, 0, 0],
                   [1, -1, 1, 0, 0, 0],
                   [1, 1, -1, 0, 0, 0],
                   [0, 0, 0, -4, 0, 0],
                   [0, 0, 0, 0, -4, 0],
                   [0, 0, 0, 0, 0, -4]])

    # Eqn 11
    S = np.matmul(D, D.T)
    S11 = S[:6, :6]
    S12 = S[:6, 6:]
    S21 = S[6:, :6]
    S22 = S[6:, 6:]

    # Eqn 15, find eigenvalue and vector
    # Since S is symmetric, S12.T = S21
    tmp = np.matmul(np.linalg.inv(C1), S11 - np.matmul(S12, np.matmul(np.linalg.inv(S22), S21)))
    eigen_value, eigen_vector = np.linalg.eig(tmp)
    u1 = eigen_vector[:, np.argmax(eigen_value)]

    # Eqn 13 solution
    u2 = np.matmul(-np.matmul(np.linalg.inv(S22), S21), u1)

    # Total solution
    u = np.concatenate([u1, u2]).T

    q = np.array([[u[0], u[5], u[4]],
                  [u[5], u[1], u[3]],
                  [u[4], u[3], u[2]]])

    n = np.array([[u[6]],
                  [u[7]],
                  [u[8]]])

    d = u[9]

    return q, n, d


def calibration_reference_magnetometer(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    q, n, d = fit_ellipsoid(x, y, z)
    q_inv = np.linalg.inv(q)
    b = -np.dot(q_inv, n)
    a_inv = np.real(1 / np.sqrt(np.dot(n.T, np.dot(q_inv, n)) - d) * linalg.sqrtm(q))
    return a_inv, b


def apply_transformation(data, a_inv, b):
    h = np.array([data]).T
    h_hat = np.matmul(a_inv, h - b)
    return h_hat


def apply_transformation_to_dataset(data, a_inv, b):
    transform_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        transform_data[i] = apply_transformation(data[i].tolist(), a_inv, b).T
    return transform_data


# linear lsm

def solve_linear_lsm(data1, d):
    a1 = data1[:, 0]
    a2 = data1[:, 1]
    a3 = data1[:, 2]
    b = np.ones(a1.shape)
    h1 = np.array([a1, a2, a3, b])
    h2 = np.matmul(h1, h1.T)
    h2_inv = np.linalg.inv(h2)
    h3 = np.matmul(h1.T, h2_inv)
    res = np.matmul(d, h3)
    return res


def get_linear_transformation(data, data_ref):
    tx = solve_linear_lsm(data, data_ref[:, 0])
    ty = solve_linear_lsm(data, data_ref[:, 1])
    tz = solve_linear_lsm(data, data_ref[:, 2])
    return np.array([tx, ty, tz])


def transform_data_linear(src, transform_array):
    res = np.zeros(src.shape)
    for i in range(src.shape[0]):
        res[i, :] = np.matmul(src[i, :], transform_array[:, :3]) + transform_array[:, 3]
    return res
