import numpy as np
from numpy import linalg as LA
from Lorentz_manifold import *


def HPA_align_tan(data_set):
    num_batch = len(data_set)
    num_dim   = data_set[0].shape[1]
    mu_       = np.zeros([num_batch, num_dim])
    var_      = np.zeros([num_batch, 1])

    for ii in range(num_batch):
        mu_[ii, :] = lorentz_Frechet_mean(data_set[ii]).numpy()
        var_[ii]   = compute_var(mu_[ii, :], data_set[ii]).numpy()

    mu = lorentz_Frechet_mean(mu_).numpy()
    PT_Scale_rotate_data = []

    for jj in range(num_batch):
        temp     = pt_scale_data_stay_tangent(data_set[jj], mu_[jj, :], mu, var_[jj])
        temp_sub = temp[:, 1:] - np.mean(temp[:,1:] , axis = 0)
        u, _, _  = LA.svd(temp_sub.T)

        temp_return = np.zeros(temp.shape)
        if jj == 0:
            u1 = u
            temp_rotate = (u1.T @ temp_sub.T).T + np.mean(temp[:, 1:], axis = 0)
        else:
            for kk in range(num_dim-1):
                u[:, kk] = np.sign(np.inner(u1[:,kk], u[:,kk])) * u[:,kk]

            temp_rotate = (u.T @ temp_sub.T).T + np.mean(temp[:, 1:], axis = 0)

        temp_return[:, 1:] = temp_rotate
        temp_return[:,0]   = (temp_rotate @ mu[1:])/mu[0]

        temp_mani = exp_data(temp_return, mu)
        PT_Scale_rotate_data.append(temp_mani)

    return PT_Scale_rotate_data


def HPA_align_tan_2set(data_set):
    num_batch = len(data_set)
    num_dim = data_set[0].shape[1]
    mu_ = np.zeros([num_batch, num_dim])
    var_ = np.zeros([num_batch, 1])

    for ii in range(num_batch):
        mu_[ii, :] = lorentz_Frechet_mean(data_set[ii]).numpy()
        var_[ii] = compute_var(mu_[ii, :], data_set[ii])

    mu = lorentz_Frechet_mean(mu_).numpy()
    PT_Scale_rotate_data = []

    for jj in range(num_batch):
        if jj == 0:
            temp = pt_scale_data_stay_tangent_2sets(data_set[jj], mu_[jj, :], mu)
        else:
            temp = pt_scale_data_stay_tangent_2sets(data_set[jj], mu_[jj, :], mu, var_[jj], var_[jj - 1])
        temp_sub = temp[:, 1:] - np.mean(temp[:, 1:], axis=0)
        u, _, _ = LA.svd(temp_sub.T)

        temp_return = np.zeros(temp.shape)
        if jj == 0:
            u1 = u
            temp_rotate = (u1.T @ temp_sub.T).T + np.mean(temp[:, 1:], axis=0)
        else:
            for kk in range(num_dim - 1):
                u[:, kk] = np.sign(np.inner(u1[:, kk], u[:, kk])) * u[:, kk]

            temp_rotate = (u.T @ temp_sub.T).T + np.mean(temp[:, 1:], axis=0)

        temp_return[:, 1:] = temp_rotate
        temp_return[:, 0] = (temp_rotate @ mu[1:]) / mu[0]

        temp_mani = exp_data(temp_return, mu)
        PT_Scale_rotate_data.append(temp_mani)

    return PT_Scale_rotate_data


def just_pt(data_set):
    num_batch = len(data_set)
    num_dim = data_set[0].shape[1]
    mu_ = np.zeros([num_batch, num_dim])
    var_ = np.zeros([num_batch, 1])
    for ii in range(num_batch):
        mu_[ii, :] = lorentz_Frechet_mean(data_set[ii]).numpy()

    mu = lorentz_Frechet_mean(mu_).numpy()
    PT_data = []
    for jj in range(num_batch):
        temp = pt_data(data_set[jj], mu_[jj, :], mu)
        PT_data.append(temp)

    return PT_data



def lorentz_Frechet_mean(set_x):
    """"
    implemented based on
    A. Lou, I. Katsman, Q. Jiang, S. Belongie, S.-N. Lim, and C. De Sa. Differentiating through the fréchet mean.
    ICML 2020
    Adapted the stopping criteria with geodesic distance
    """
    set_x = torch.from_numpy(set_x).double()
    manifold = Lorentz_m()

    y, u = [], []
    y.append(torch.from_numpy(np.hstack([np.ones(1), np.zeros(set_x.shape[1]-1)])).double())
    eps = 1
    while(eps>1e-6):
        tmp = 0
        for i in range(set_x.shape[0]):
            tt_input = manifold.lorentz_inner_product_v(-set_x[i,:], y[-1])
            if ((tt_input.pow(2)-1)< 1e-6):
                tmp = tmp
            else:
                aa = (2 * arcosh(tt_input) / ((tt_input.pow(2) - 1).sqrt())) * set_x[i,:]
                tmp = tmp + aa
        y.append(tmp / (manifold.lorentz_inner_product_v(-tmp, tmp).sqrt()))
        eps = pair_wise_lorentz_dis_point(y[-2],y[-1]) ** 2

        # print('eps: ', eps)
        # print('y: ', y[-1])

    return y[-1]

def geo_path_data(data, m_point):
    data     = torch.from_numpy(data)
    m_point  = torch.from_numpy(m_point)
    scale    = torch.from_numpy(np.random.rand(1)) + 0.5
    manifold = Lorentz_m()

    geo_path_data_points = np.zeros(data.shape)
    for i in range(data.shape[0]):
        tan_p   = manifold.Log_Map(m_point, data[i, :])
        scale_p = torch.div(tan_p, scale)
        test_p  = manifold.Exp_Map(m_point, scale_p)
        geo_path_data_points[i, :] = test_p

    return geo_path_data_points


def pt_data_from_tan(data, mean_loc, mean_glo):
    data = torch.from_numpy(data)
    mean_loc = torch.from_numpy(mean_loc)
    mean_glo = torch.from_numpy(mean_glo)

    manifold = Lorentz_m()
    transported_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        pt_p = manifold.PT(mean_loc, mean_glo, data[i, :])
        test_p = manifold.Exp_Map(mean_glo, pt_p)

        transported_data[i, :] = test_p
    return transported_data


def pt_data(data, mean_loc, mean_glo):
    data = torch.from_numpy(data)
    mean_loc = torch.from_numpy(mean_loc)
    mean_glo = torch.from_numpy(mean_glo)

    for i in range(data.shape[0]):
        data[i,:] = data[i,:] - beta(data[i,:], mean_loc, mean_glo) * mean_loc +\
                    gamma(data[i,:], mean_loc, mean_glo) * mean_glo

    return data


def beta(data, mean_loc, mean_glo):
    manifold = Lorentz_m()
    alpha    = - manifold.lorentz_inner_product_v(mean_loc, mean_glo)
    return - manifold.lorentz_inner_product_v(data, (mean_loc+ mean_glo)/(alpha+1))


def gamma(data, mean_loc, mean_glo):
    manifold = Lorentz_m()
    alpha    = - manifold.lorentz_inner_product_v(mean_loc, mean_glo)
    return manifold.lorentz_inner_product_v(data, (mean_glo - (2 * alpha) * mean_loc)/(alpha+1))


def pt_data_proj(data, mean_loc, mean_glo):
    data = torch.from_numpy(data)
    mean_loc = torch.from_numpy(mean_loc)
    mean_glo = torch.from_numpy(mean_glo)

    manifold = Lorentz_m()
    transported_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        tan_p = manifold.Log_Map(mean_loc, data[i, :])
        pt_p = manifold.PT(mean_loc, mean_glo, tan_p)
        test_p = manifold.Exp_Map(mean_glo, pt_p)

        transported_data[i, :] = test_p

    return transported_data


def pt_scale_data(data, mean_loc, mean_glo, var):
    data = torch.from_numpy(data)
    mean_loc = torch.from_numpy(mean_loc)
    mean_glo = torch.from_numpy(mean_glo)
    var = torch.from_numpy(var)

    manifold = Lorentz_m()
    transported_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        tan_p = manifold.Log_Map(mean_loc, data[i, :])
        pt_p = torch.div(manifold.PT(mean_loc, mean_glo, tan_p), np.sqrt(var))
        test_p = manifold.Exp_Map(mean_glo, pt_p)

        transported_data[i, :] = test_p

    return transported_data


def pt_scale_data_stay_tangent(data, mean_loc, mean_glo, var):
    data     = torch.from_numpy(data)
    mean_loc = torch.from_numpy(mean_loc)
    mean_glo = torch.from_numpy(mean_glo)
    var      = torch.from_numpy(var)

    manifold = Lorentz_m()
    transported_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        tan_p  = manifold.Log_Map(mean_loc, data[i,:])
        pt_p   = torch.div(manifold.transp(mean_loc, mean_glo, tan_p) , np.sqrt(var))

        transported_data[i,:] = pt_p

    return transported_data


def pt_scale_data_stay_tangent_2sets(data, mean_loc, mean_glo, var1=None, var2=None):
    data = torch.from_numpy(data)
    mean_loc = torch.from_numpy(mean_loc)
    mean_glo = torch.from_numpy(mean_glo)
    if var1 is not None:
        var1 = torch.from_numpy(var1)
        var2 = torch.from_numpy(var2)

    manifold = Lorentz_m()
    transported_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        tan_p = manifold.Log_Map(mean_loc, data[i, :])
        if var1 is None and var2 is None:
            pt_p = manifold.PT(mean_loc, mean_glo, tan_p)
        else:
            pt_p = torch.mul(manifold.PT(mean_loc, mean_glo, tan_p), np.sqrt(var2 / var1))

        transported_data[i, :] = pt_p

    return transported_data


def compute_var(mean, data):
    distance = pair_wise_lorentz_dis(data, mean)
    return np.sum(distance) / data.shape[0]


def exp_data(data, mean):
    data = torch.from_numpy(data)
    mean = torch.from_numpy(mean)

    manifold = Lorentz_m()
    returned_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        test_p = manifold.Exp_Map(mean, data[i, :])

        returned_data[i, :] = test_p

    return returned_data


def pair_wise_lorentz_dis(x, y):
    size_n   = x.shape[1]
    M        = np.eye(size_n)
    M[0, 0]  = -1
    TT       = - x @ M @ y.T
    TT       = np.where(TT > 1, TT, 1 + 1e-12)
    return np.arccosh(TT)


def pair_wise_lorentz_dis_point(vector_x, vector_y):
    manifold = Lorentz_m()
    xy_lorentz = - manifold.lorentz_inner_product_v(vector_x, vector_y)
    xy_lorentz = np.where(xy_lorentz > 1, xy_lorentz, 1 + 1e-12)
    return np.arccosh(xy_lorentz)
