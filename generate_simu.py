from numpy import *
from scipy.stats import ortho_group
from scipy.linalg import sqrtm
from manifold_func import *


def generate_dataset_simu1(num_dataset, ref_center, num_realization):
    data = []
    dim = ref_center.shape[0]
    mu_0 = np.zeros(dim)
    mu_0[0] = 1

    Cov_mat = np.cov(np.random.rand(dim - 1, dim - 1))
    S1_p = np.random.multivariate_normal(np.zeros(dim - 1), Cov_mat, num_realization)
    S1 = np.concatenate((np.zeros([num_realization, 1]), S1_p), axis=1)
    S1_ = pt_data_from_tan(S1, mu_0, ref_center)

    data.append(S1_)
    for ii in range(num_dataset - 1):
        L = randn_lorentz_transform(dim - 1, 2)
        S = np.zeros(S1_.shape)

        for jj in range(num_realization):
            T = randn_lorentz_translation(dim - 1)
            LT = real(np.array(L @ T))
            S[jj, :] = (LT @ S1_[jj, :].T).T
            if S[jj, 0] < 0:
                S[jj, 0] = - S[jj, 0]
        data.append(S)

    return data


def generate_dataset_simu2(num_dataset, ref_center, num_realization):

    data    = []

    dim     = ref_center.shape[0]
    mu_0    = np.zeros(dim)
    mu_0[0] = 1

    Cov_mat = np.cov(np.random.rand(dim-1, dim-1))
    S1_p    = np.random.multivariate_normal(np.zeros(dim-1), Cov_mat, num_realization)
    S1      = np.concatenate((np.zeros([num_realization, 1]), S1_p), axis = 1)
    S1_     = pt_data_from_tan(S1, mu_0, ref_center)

    data.append(S1_)
    for ii in range(num_dataset-1):
        L = randn_lorentz_transform(dim-1, 2)
        S = np.concatenate((np.zeros([num_realization, 1]), S1_p +
                     np.random.multivariate_normal(np.zeros(dim-1), 1 * np.eye(dim-1), num_realization)
                    ), axis = 1)
        data.append(real(L @ pt_data_from_tan(S, mu_0, ref_center).T).T)

    return data


def generate_dataset_simu3(num_dataset, ref_center, num_realization, radius):

    data    = []

    dim     = ref_center.shape[0]
    mu_0    = np.zeros(dim)
    mu_0[0] = 1

    Cov_mat = np.cov(np.random.rand(dim-1, dim-1))
    S1_p    = np.random.multivariate_normal(np.zeros(dim-1), Cov_mat, num_realization)
    S1      = np.concatenate((np.zeros([num_realization, 1]), S1_p), axis = 1)
    S1_     = pt_data_from_tan(S1, mu_0, ref_center)


    data.append(S1_)
    for ii in range(num_dataset-1):
        set_mean = generate_high_dim_lorentz_point(dim, radius, 1)
        L = randn_lorentz_transform(dim-1, 2)
        S = np.concatenate((np.zeros([num_realization, 1]), S1_p +
                     np.random.multivariate_normal(np.zeros(dim-1), 1 * np.eye(dim-1), num_realization)
                    ), axis = 1)

        S_tmp = real(L @ pt_data_from_tan(S, mu_0, ref_center).T).T
        data.append(geo_path_data(S_tmp, set_mean[0]))

    return data


def randn_lorentz_transform(size_n, k):
    U = np.eye(size_n + 1)

    for kk in range(k):
        check = np.random.rand(1)
        if check > 0.5:
            O = ortho_group.rvs(size_n)
            R_U = np.block([
                [1, np.zeros((1, size_n))],
                [np.zeros((size_n, 1)), O]
            ])
            U = R_U @ U
        else:
            v = np.random.rand(size_n, 1)
            T = sqrtm(np.eye(size_n) + v @ v.T)
            R_T = np.block([
                [np.sqrt(1 + LA.norm(v) ** 2), v.T],
                [v, T]
            ])
            U = R_T @ U

    return U


def randn_lorentz_translation(size_n):
    v1 = np.matrix(np.random.normal(0, 1, size_n))
    T = sqrtm(np.eye(size_n) + v1.T @ v1)
    R_T = np.block([
        [np.sqrt(1 + LA.norm(v1) ** 2), v1],
        [v1.T, T]
    ])

    return R_T


def generate_high_dim_lorentz_point(dim, range_size, num_realization):
    data = np.zeros([num_realization, dim])
    for ii in range(num_realization):
        data[ii, 1:] = (np.random.random_sample((dim - 1,)) - 0.5) * 2 * range_size
        data[ii, 0] = np.sqrt(1 + np.inner(data[ii, 1:], data[ii, 1:]))
    return data


def lorentz_translation(vector):
    vector = np.matrix(vector)
    T = sqrtm(np.eye(vector.shape[1]) + vector.T @ vector)
    R_T = np.block([
        [np.sqrt(1 + LA.norm(vector) ** 2), vector],
        [vector.T, T]
    ])

    return real(R_T)


def lorentz_rotation(input_matrix):
    size_n = input_matrix.shape[0]
    R_U = np.block([
        [1, np.zeros((1, size_n))],
        [np.zeros((size_n, 1)), input_matrix]
    ])
    return R_U



