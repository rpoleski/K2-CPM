import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from prfdata import PrfData
from campaigngridradec2pix import CampaignGridRaDec2Pix
from prfforcampaign import PrfForCampaign
import cpm_part1
import cpm_part2
import matrix_xy
import plot_utils
import tpfdata


# This example illustrates usage of cpm_part2_prf_model_flux() - a function 
# for simultanuous model fitting and photometry extraction. To see the main 
# part of the code, please go to "if __name__ == ..." command below
def transform_model(t_0, amplitude_ratio, width_ratio, model_dt, model_flux, time):
    """Simple function for scaling and linear interpolation.
    First 3 parameters are floats, the rest are vectors.
    """
    model_time = t_0 + model_dt * width_ratio
    model = np.interp(time, model_time, model_flux) * amplitude_ratio
    return model
    
def cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask,
        prfs, mask_prfs, model, l2, n_pix):
    """runs CPM on a set of pixels and returns each result"""
    out_signal = []
    out_mask = []
    for i in range(n_pix):
        (signal, mask_cpm) = cpm_part2.cpm_part2_prf_model_flux(
                            tpf_flux[i], None, tpf_epoch_mask,
                            predictor_matrix, predictor_mask,
                            prfs[:,i], mask_prfs, model, 
                            l2=l2)
        out_signal.append(signal)
        out_mask.append(mask_cpm)
    return (out_signal, out_mask)

def mean_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask,
        prfs, mask_prfs, model, l2, n_pix):
    """runs CPM on a set of pixels and returns mean residue"""
    (signal, mask) = cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
        predictor_mask, prfs, mask_prfs, model, l2, n_pix)
    sum_out = 0.
    sum_n = 0
    for i in range(n_pix):
        sum_out += np.sum(signal[i][mask[i]]**2) 
        sum_n += np.sum(mask[i])
    return (sum_out / sum_n)**0.5
    
def summed_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, predictor_mask,
        prfs, mask_prfs, model, l2, n_pix):
    """runs CPM and adds all the results"""
    (signal, mask) = cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
        predictor_mask, prfs, mask_prfs, model, l2, n_pix)
    out = signal[0] * 0.
    for i in range(n_pix):
        out[mask[i]] += signal[i][mask[i]]
    return (out, mask[0])

def fun_1(inputs, model_dt, model_flux, time, tpf_flux, tpf_epoch_mask, 
        predictor_matrix, predictor_mask, prfs, mask_prfs, l2, n_pix):
    """1-parameter function for optimisation"""
    t_0 = inputs[0]

    model = transform_model(t_0, 1., 1., model_dt, model_flux, time)

    out = mean_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
            predictor_mask, prfs, mask_prfs, model, l2, n_pix)

    return out

def fun_2(inputs, model_dt, model_flux, time, tpf_flux, tpf_epoch_mask, 
        predictor_matrix, predictor_mask, prfs, mask_prfs, l2, n_pix):
    """2-parameter function for optimisation"""
    t_0 = inputs[0]
    amplitude_factor = inputs[1]

    model = transform_model(t_0, amplitude_factor, 1., model_dt, model_flux, 
                            time)

    out = mean_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
            predictor_mask, prfs, mask_prfs, model, l2, n_pix)

    return out

def fun_3(inputs, model_dt, model_flux, time, tpf_flux, tpf_epoch_mask, 
        predictor_matrix, predictor_mask, prfs, mask_prfs, l2, n_pix):
    """3-parameter function for optimisation"""
    t_0 = inputs[0]
    amplitude_factor = inputs[1]
    width_ratio = inputs[2]

    model = transform_model(t_0, amplitude_factor, width_ratio, model_dt, 
                            model_flux, time)

    out = mean_cpm_output(tpf_flux, tpf_epoch_mask, predictor_matrix, 
            predictor_mask, prfs, mask_prfs, model, l2, n_pix)

    return out

def finish_figure(file_name, title=None, xlabel=None, legend=True):
    """some standard things to be done after the main plot is done"""
    if legend:
        plt.legend(loc='lower left')
    if xlabel is None:
        plt.xlabel('BJD-245000')
    else:
        plt.xlabel(xlabel)
    plt.ylabel('relative flux')
    if title is not None:
        plt.title(title)
    plt.savefig(file_name)
    plt.close()
    plt.gca().set_prop_cycle(None)

def plot_pixel_curves(pixels, fluxes, time, file_name):
    """construct a matrix using list and plot it"""
    fig = plt.gcf()
    fig.set_size_inches(50, 30)
    
    flux_matrix = plot_utils.construct_matrix_from_list(pixels, fluxes)
    plot_utils.plot_matrix_subplots(fig, time, flux_matrix)

    plt.savefig(file_name)
    plt.close()

def mean_position_clipped(grids, ra, dec, radius=2):
    """get mean position with input data narrowed down to radius"""
    (pixel_x, pixel_y) = grids.apply_grids(ra, dec)
    weights = grids.sigma[grids.mask]**-2
    mean_x = (pixel_x[grids.mask] * weights).sum() / weights.sum()
    mean_y = (pixel_y[grids.mask] * weights).sum() / weights.sum()
    mask = ((pixel_x[grids.mask]-mean_x)**2 + (pixel_y[grids.mask]-mean_y)**2 < radius**2)
    weights = weights[mask]
    mean_x = (pixel_x[grids.mask][mask] * weights).sum() / weights.sum()
    mean_y = (pixel_y[grids.mask][mask] * weights).sum() / weights.sum()
    mask_out = np.copy(grids.mask)
    mask_out[grids.mask] = mask
    return (mean_x, mean_y, mask_out)


if __name__ == "__main__":
    # We want to extract the light curve of OGLE-BLG-ECL-234840, which is
    # an eclipsing binary with a single eclipse in subcampaign 91. I know 
    # the star coordinates and that it is in channel 31, but you can find 
    # the channel this way:
    # >>> import K2fov
    # >>> K2fov.fields.getKeplerFov(9)pickAChannel(ra, dec)
    channel = 31
    campaign = 91
    ra = 269.929125
    dec = -28.410833

    # Now we need to know the average position of the target. For this we 
    # need to setup grid transformations:
    grids = CampaignGridRaDec2Pix(campaign=campaign, channel=channel)
    # and then get mean position:
    (mean_x, mean_y, grids_mask) = mean_position_clipped(grids, ra, dec)
    print("Mean target position: {:.2f} {:.2f}\n".format(mean_x, mean_y))

    # Remove some epochs that we know are bad:
    for i in [510, 860, 856, 1004, 968]:
    #for i in [510, 860, 856, 1004, 968, 475, 474]:
        grids_mask[i] = False

    # Now we want to select a few pixels with most flux. First, let's make 
    # temporary list of pixels, let's say 5x5 i.e., "half_size" of 2:
    half_size = 2
    pixels = matrix_xy.pixel_list_center(mean_x, mean_y, half_size)

    # Second, setup the PRF directory:
    prf_dir = "./PRF_files"
    PrfData.data_directory = prf_dir
    prf_template = PrfData(channel=channel)
    # Third, the highest level structure - something that combines grids and 
    # PRF data:
    prf_for_campaign = PrfForCampaign(campaign=campaign, grids=grids, 
                                    prf_data=prf_template)
    # Now the important step - getting prf value for every pixel and every epoch
    (failed_prfs, mask_prfs, prfs) = prf_for_campaign.apply_grids_and_prf(ra, 
                                                                dec, pixels)
    mask_prfs *= grids_mask

    # Fourth, for each pixel sum all PRF values: 
    prf_sum = np.sum(prfs[mask_prfs], axis=0)
    
    # Then let's sort and select top 20:
    n_select = 20
    sorted_indexes = np.argsort(prf_sum)[::-1][:n_select]
    
    # We have that, so lets get rid of all the rest of the pixels:
    pixels = pixels[sorted_indexes]
    prfs = prfs[:, sorted_indexes]

    print("Now we run CPM part 1.") 
    # We need a number of settings:
    n_predictor = 400
    n_pca = 0
    distance = 16
    exclusion = 1
    flux_lim = (0.2, 2.0)
    tpf_dir = "tpf/" # In particular, the path to tpf data has to be given.
    file_1_name = "eb234840_predictor_{:}.dat".format(n_predictor)
    file_2_name = "eb234840_predictor_mask_{:}.dat".format(n_predictor)
    if path.isfile(file_1_name) and path.isfile(file_2_name):
        predictor_matrix = matrix_xy.load_matrix_xy(file_1_name)
        predictor_mask = cpm_part2.read_true_false_file(file_2_name)
    else: # This is done only once after n_predictor was changed
    # We in fact run it only for a single central pixel (see 
    # pixel_list=np.array([pixels[0][::-1]]) ), it won't affect results and 
    # makes life easier.
        print("This may take a while...")
        (predictor_matrix, predictor_mask) = cpm_part1.run_cpm_part1(
            channel=channel, campaign=campaign, num_predictor=n_predictor,
            num_pca=n_pca, dis=distance, excl=exclusion, flux_lim=flux_lim, 
            input_dir=tpf_dir, pixel_list=np.array([pixels[0][::-1]]), 
            train_lim=None, return_predictor_epoch_masks=True)
        matrix_xy.save_matrix_xy(predictor_matrix[0], file_1_name)
        np.savetxt(file_2_name, predictor_mask[0], fmt='%r')
        predictor_matrix = predictor_matrix[0]
        predictor_mask = predictor_mask[0]
        print("Done!\n")

    # Now extract raw light curves from TPF files:
    epic_number = '200069761' # in case you don't know it, run:
    # >>> from K2CPM import wcsfromtpf
    # >>> epic_number = wcsfromtpf.WcsFromTpf(channel, campaign).get_nearest_pixel_radec(ra, dec)[4]
    tpfdata.TpfData.directory = tpf_dir
    tpf = tpfdata.TpfData(epic_id=epic_number, campaign=campaign)
    tpf_flux = tpf.get_fluxes_for_pixel_list(pixels)

    # Now we read data with astrophysical model:
    model_file = "example_1_model_averaged.dat"
    (model_dt, model_flux) = np.loadtxt(model_file, unpack=True)
    model_flux[model_dt < -13.] = 0.
    model_flux[model_dt > 13.] = 0.
    
    # common fit settings:
    l2 = 10**8.5
    n_pix_use = 10
    tol = 0.01
    args = (model_dt, model_flux, tpf.jd_short, tpf_flux, tpf.epoch_mask, 
            predictor_matrix, predictor_mask, prfs, mask_prfs, l2, n_pix_use)
    file_plot_1 = "example_4_plot_1.png"
    file_plot_2 = "example_4_plot_2.png"
    file_plot_3 = "example_4_plot_3.png"
    
    # simple 1-D fit:
    start = np.array([7515.])
    out = minimize(fun_1, start, args=args, tol=tol)
    print()
    print(out)
    
    # plot 1-D fit:
    model = transform_model(out.x[0], 1., 1., model_dt, model_flux, tpf.jd_short)
    (result, result_mask) = summed_cpm_output(tpf_flux, tpf.epoch_mask, 
        predictor_matrix, predictor_mask, prfs, mask_prfs, model, l2, n_pix_use)    
    plt.plot(tpf.jd_short[result_mask], model[result_mask]+result[result_mask], '.')
    plt.plot(tpf.jd_short[result_mask], model[result_mask], '-')
    finish_figure(file_plot_1, title=None, legend=False)
    #worst results:
    #print(np.argsort(np.abs(result))[-5:])
    #print(result[np.argsort(np.abs(result))[-5:]])
    
    # simple 2-D fit:
    start = np.array([7515., 2.0])
    out = minimize(fun_2, start, args=args, tol=tol)
    print()
    print(out)

    # plot 2-D fit:
    model = transform_model(out.x[0], out.x[1], 1., model_dt, model_flux, tpf.jd_short)
    (result, result_mask) = summed_cpm_output(tpf_flux, tpf.epoch_mask, 
        predictor_matrix, predictor_mask, prfs, mask_prfs, model, l2, n_pix_use)    
    plt.plot(tpf.jd_short[result_mask], model[result_mask]+result[result_mask], '.')
    plt.plot(tpf.jd_short[result_mask], model[result_mask], '-')
    finish_figure(file_plot_2, title=None, legend=False)

    # simple 3-D fit:
    start = np.array([7517., 0.5, 0.3])
    #bounds = ((None, None), (0., None), (0.2, 5.0))
    out = minimize(fun_3, start, args=args, tol=tol)#, bounds=bounds)
    print()
    print(out)

    # plot 3-D fit:
    model = transform_model(out.x[0], out.x[1], out.x[2], model_dt, model_flux, tpf.jd_short)
    (result, result_mask) = summed_cpm_output(tpf_flux, tpf.epoch_mask, 
        predictor_matrix, predictor_mask, prfs, mask_prfs, model, l2, n_pix_use)    
    plt.plot(tpf.jd_short[result_mask], model[result_mask]+result[result_mask], '.')
    plt.plot(tpf.jd_short[result_mask], model[result_mask], '-')
    finish_figure(file_plot_3, title=None, legend=False)
