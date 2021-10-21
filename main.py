from generate_simu import *

if __name__ == '__main__':
    seed            = 9512
    trial           = 10
    num_realization = 100
    radius          = 5
    dim_list_input  = np.array([ 3, 5, 10, 20, 30, 40])
    len_list        = dim_list_input.shape[0]
    Discrepancy_mat = np.zeros([len_list, 6, trial])

    random.seed(seed)
    np.random.seed(seed)

    for ii in range(len_list):
        print('dim = ', dim_list_input[ii])
        for jj in range(trial):
            print('trial = ', jj)
            tmp       = generate_high_dim_lorentz_point(dim_list_input[ii], radius, 1)
            test_data_set1 = generate_dataset_simu1(2, tmp[0,:], num_realization)
            test_data_set2 = generate_dataset_simu2(2, tmp[0,:], num_realization)
            test_data_set3 = generate_dataset_simu3(2, tmp[0,:], num_realization, radius)

            hpa_data_set1  = HPA_align_tan_2set(test_data_set1)
            hpa_data_set2  = HPA_align_tan_2set(test_data_set2)
            hpa_data_set3  = HPA_align_tan_2set(test_data_set3)

            Discrepancy_mat[ii, 0, jj] = np.mean(np.diag(pair_wise_lorentz_dis(test_data_set1[0], test_data_set1[1]))**2)
            Discrepancy_mat[ii, 1, jj] = np.mean(np.diag(pair_wise_lorentz_dis(hpa_data_set1[0], hpa_data_set1[1]))**2)
            Discrepancy_mat[ii, 2, jj] = np.mean(np.diag(pair_wise_lorentz_dis(test_data_set2[0], test_data_set2[1]))**2)
            Discrepancy_mat[ii, 3, jj] = np.mean(np.diag(pair_wise_lorentz_dis(hpa_data_set2[0], hpa_data_set2[1]))**2)
            Discrepancy_mat[ii, 4, jj] = np.mean(np.diag(pair_wise_lorentz_dis(test_data_set3[0], test_data_set3[1]))**2)
            Discrepancy_mat[ii, 5, jj] = np.mean(np.diag(pair_wise_lorentz_dis(hpa_data_set3[0], hpa_data_set3[1]))**2)

    print('Discrepancy of Q1-Q2: ', np.mean(Discrepancy_mat[:,0,:], axis = 1))
    print('After HPA: ',np.mean(Discrepancy_mat[:,1,:], axis = 1))
    print('Discrepancy of Q1-Q3: ', np.mean(Discrepancy_mat[:,2,:], axis = 1))
    print('After HPA: ',np.mean(Discrepancy_mat[:,3,:], axis = 1))
    print('Discrepancy of Q1-Q4: ', np.mean(Discrepancy_mat[:,4,:], axis = 1))
    print('After HPA: ', np.mean(Discrepancy_mat[:,5,:], axis = 1))

