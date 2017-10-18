import sys
import numpy as np

from K2CPM import k2_cpm_small
from K2CPM import matrix_xy
from K2CPM import leastSquareSolver as lss


def read_true_false_file(file_name):
    """reads file with values True or False"""
    parser = {'TRUE': True, 'FALSE': False}
    out = []
    with open(file_name) as in_file:
        for line in in_file.readlines():
            out.append(parser[line[:-1].upper()])
    return np.array(out)

def check_l2_and_l2_per_pixel(l2, l2_per_pixel):
    """check if exactly one of l2 and l2_per_pixel is set to a float value"""
    if (l2 is None) == (l2_per_pixel is None):
        raise ValueError('In cpm_part2 you must set either l2 or l2_per_pixel')
    if l2_per_pixel is not None:
        if not isinstance(l2_per_pixel, (float, np.floating)):
            raise TypeError('l2_per_pixel must be of float type')
    else:
        if not isinstance(l2, (float, np.floating)):
            raise TypeError('l2 must be of float type')

def cpm_part2_subcampaigns(tpf_time_train, tpf_flux_train, tpf_flux_err_train,
        tpf_epoch_mask_train, predictor_matrix_train, predictor_mask_train,
        tpf_time_apply, tpf_flux_apply, tpf_flux_err_apply, 
        tpf_epoch_mask_apply, predictor_matrix_apply, predictor_mask_apply, 
        l2=None, l2_per_pixel=None, train_lim=None, model=None):
    """This version of CPM part2 find coefficients using one subcampaign and 
    applies them to the other subcampaign."""
    check_l2_and_l2_per_pixel(l2, l2_per_pixel)
    if l2_per_pixel is not None:
        l2 = l2_per_pixel * predictor_matrix_train.shape[1]

    fit_matrix_results_1 = k2_cpm_small.get_fit_matrix_ffi(tpf_flux_train, tpf_epoch_mask_train, predictor_matrix_train, predictor_mask_train, l2, tpf_time_train, poly=0, ml=model)
    (target_flux_train, predictor_matrix_train, _, l2_vector_train, time_train) = fit_matrix_results_1

    fit_matrix_results_2 = k2_cpm_small.get_fit_matrix_ffi(tpf_flux_apply, tpf_epoch_mask_apply, predictor_matrix_apply, predictor_mask_apply, l2, tpf_time_apply, poly=0, ml=model)
    (target_flux_apply, predictor_matrix_apply, _, _, time_apply) = fit_matrix_results_2

    result = k2_cpm_small.fit_target(target_flux_train, np.copy(predictor_matrix_train), l2_vector=l2_vector_train, train_lim=train_lim, time=time_train)

    fit_flux = np.dot(predictor_matrix_apply, result)
    dif = target_flux_apply - fit_flux[:,0]
    return (result, fit_flux, dif, time_apply)

def cpm_part2_prf_model_flux(tpf_flux, tpf_flux_err, tpf_flux_mask,
            predictor_matrix, predictor_matrix_mask,
            prf, prf_mask,
            model_flux,
            l2=None, l2_per_pixel=None, 
            train_lim=None, time=None,
            add_constant=False):
    """
    Run part 2 of CPM knowing the PRF values for given pixel and the model 
    of total flux from given source (in the case of microlensing F_s*A). 
    All the required parameters have the same length of the first dimension. 

    tpf_flux_err is not used currently
    """
    check_l2_and_l2_per_pixel(l2, l2_per_pixel)
    if l2_per_pixel is not None:
        l2 = l2_per_pixel * predictor_matrix.shape[1]

    mask = tpf_flux_mask * predictor_matrix_mask * prf_mask
    if np.any(np.isnan(model_flux[mask])):
        raise ValueError('unmasked nan values in model input for cpm_part2_prf_model_flux()')
    predictor_masked = predictor_matrix[mask]
    flux_masked = tpf_flux[mask] - prf[mask] * model_flux[mask]

    if add_constant:
        adds = np.ones(predictor_masked.shape[0]) * np.mean(flux_masked)
        predictor_masked = np.concatenate((predictor_masked, adds), axis=1)

    if train_lim is not None:
        if time is None:
            raise ValueError('parameter time has to be provided when train_lim parameter is provided')
        train_mask = (time[mask]<train_lim[0]) | (time[mask]>train_lim[1])
    else:
        train_mask = np.ones(sum(mask),  dtype=bool)

    covar = None
    coefs = lss.linear_least_squares(predictor_masked[train_mask], flux_masked[train_mask], covar, l2)

    fit_flux = np.dot(predictor_masked, coefs)

    residue = flux_masked - fit_flux[:,0]
    residue_out = np.zeros(len(mask), dtype=float)
    residue_out[mask] = residue

    return (residue_out, mask)

# TO_BE_DONE - use tpf_flux_err if user wants
def cpm_part2(tpf_time, tpf_flux, tpf_flux_err, tpf_epoch_mask, 
            predictor_matrix, predictor_mask, 
            l2=None, l2_per_pixel=None, 
            train_lim=None, model=None):
    """get predictor_matrix, run CPM, calculate dot product and difference of 
    target_flux and fit_flux

    Parameters l2 and l2_per_pixel define strength of regularization. 
    You have to set one of them. If l2_per_pixel is set, then 
    l2 = l2_per_pixel * number_of_predictor_pixels
    is calculated. Note that l2_per_pixel should be on order of a few, and 
    l2 should be on order of thousands. 
    """
    check_l2_and_l2_per_pixel(l2, l2_per_pixel)
    if l2_per_pixel is not None:
        l2 = l2_per_pixel * predictor_matrix.shape[1]
    
    # run get_fit_matrix_ffi() which mostly applies joint epoch_mask
    fit_matrix_results = k2_cpm_small.get_fit_matrix_ffi(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask, l2, tpf_time, 0, ml=model)
    #fit_matrix_results = k2_cpm_small.get_fit_matrix_ffi(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask, l2, tpf_time, None, ml=model)
    (target_flux, predictor_matrix, none_none, l2_vector, time) = fit_matrix_results

    # run CPM:
    result = k2_cpm_small.fit_target(target_flux, np.copy(predictor_matrix), l2_vector=l2_vector, train_lim=train_lim, time=time)
    
    # final calculations:
    fit_flux = np.dot(predictor_matrix, result)
    dif = target_flux - fit_flux[:,0]
    return (result, fit_flux, dif, time)


def execute_cpm_part2(n_test=1):
    # settings:
    l2 = 1000.
    campaign = 92 
    pixel = [883, 670]
    in_directory = "../tests/intermediate/"
    in_directory_2 = "../tests/intermediate/expected/"
    out_directory = "../tests/output/"
    
#    pre_matrix_file = "{:}{:}-pre_matrix_xy.dat".format(in_directory_2, n_test)
    pre_matrix_file = "{:}{:}-pre_matrix_xy.dat".format(in_directory, n_test)
    predictor_epoch_mask_file = "{:}{:}-predictor_epoch_mask.dat".format(in_directory, n_test)
    pixel_flux_file_name = '{:}{:}-pixel_flux.dat'.format(in_directory, n_test)
    epoch_mask_file_name = '{:}{:}-epoch_mask.dat'.format(in_directory, n_test)

    # output files:
    result_file_name = '{:}{:}-result.dat'.format(out_directory, n_test)
    dif_file_name = '{:}{:}-dif.dat'.format(out_directory, n_test)
    # Settings end here.
    
    # Load all required files:
    pre_matrix = matrix_xy.load_matrix_xy(pre_matrix_file)
    predictor_epoch_mask = read_true_false_file(predictor_epoch_mask_file)
    tpf_data = np.loadtxt(pixel_flux_file_name, unpack=True)
    (tpf_time, tpf_flux, tpf_flux_err) = tpf_data
    tpf_epoch_mask = read_true_false_file(epoch_mask_file_name)

    # Calculations:
    (result, fit_flux, dif, time) = cpm_part2(tpf_time, tpf_flux, tpf_flux_err,
                                    tpf_epoch_mask, pre_matrix, predictor_epoch_mask, l2) 

    # Save results:
    np.savetxt(result_file_name, result, fmt='%.8f')
    np.savetxt(dif_file_name, dif, fmt='%.8f')
    

if __name__ == '__main__':
    # case dependent settings:    
    n_test = 1
    #n_test = 2
    #n_test = 3
    #n_test = 4
    
    execute_cpm_part2(n_test)
    
