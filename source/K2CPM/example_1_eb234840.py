import sys
import numpy as np
import matplotlib.pyplot as plt

from prfdata import PrfData
from campaigngridradec2pix import CampaignGridRaDec2Pix
from prfforcampaign import PrfForCampaign
import cpm_part1
import cpm_part2
import matrix_xy
import plot_utils
import tpfdata


def running_mean_with_edges(vector, half_size):
    """calculate running mean including the edge points - note that this does not 
    conserve sum of data"""
    out = vector * 0.
    indexes = np.arange(len(vector))
    for i in indexes:
        mask = (np.abs(indexes - i) <= half_size)
        out[i] = sum(vector[mask]) / float(sum(mask))
    return out

def finish_figure(file_name, title=None):
    """some standard things to be done after the main plot is done"""
    plt.legend(loc='lower left')
    plt.xlabel('BJD-245000')
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


# Note that this example ilustrates standard CPM and CPM+PRF and additionally 
# shows how results change with a number of pixels included. 
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
    (mean_x, mean_y) = grids.mean_position(ra, dec)
    print("Mean target position: {:.2f} {:.2f}\n".format(mean_x, mean_y))

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

    # Fourth, for each pixel sum all PRF values: 
    prf_sum = np.sum(prfs, axis=0)
    
    # Then let's sort and select top 20:
    n_select = 20
    sorted_indexes = np.argsort(prf_sum)[::-1][:n_select]
    
    # We have that, so lets get rid of all the rest of the pixels:
    pixels = pixels[sorted_indexes]
    prfs = prfs[:, sorted_indexes]
    print("{:} pixels with highest signal: \n{:}\n".format(n_select, pixels))

    print("Now we run CPM part 1. This may take a while...")
    # We need a number of settings:
    n_predictor = 500
    n_pca = 0
    distance = 16
    exclusion = 1
    flux_lim = (0.2, 1.5)
    tpf_dir = "tpf/" # In particular, the path to tpf data has to be given.
    # We in fact run it only for a single central pixel (see 
    # pixel_list=np.array([pixels[0][::-1]]) ), it won't affect results and 
    # makes life easier.
    (predictor_matrix, predictor_mask) = cpm_part1.run_cpm_part1(
            channel=channel, campaign=campaign, num_predictor=n_predictor,
            num_pca=n_pca, dis=distance, excl=exclusion, flux_lim=flux_lim, 
            input_dir=tpf_dir, pixel_list=np.array([pixels[0][::-1]]), 
            train_lim=None, return_predictor_epoch_masks=True)
    print("Done!\n")

    # Now extract raw light ruves from TPF files:
    epic_number = '200069761' # in case you don't know it, run:
    # >>> from K2CPM import wcsfromtpf
    # >>> epic_number = wcsfromtpf.WcsFromTpf(channel, campaign).get_nearest_pixel_radec(ra, dec)[4]
    tpfdata.TpfData.directory = tpf_dir
    tpf = tpfdata.TpfData(epic_id=epic_number, campaign=campaign)
    tpf_flux = tpf.get_fluxes_for_pixel_list(pixels)

    print("Plot raw pixel curves:")
    file_name = "example_1_pixel_curves.png"
    plot_pixel_curves(pixels, tpf_flux, tpf.jd_short, file_name)
    print(file_name)

    # Now run cpm_part_2:
    l2 = 1.e4 # This is regularization strnegth.
    train_limits = [7508., 7540.] # We train on data before 2457508.
    ok = ((tpf.jd_short[np.isfinite(tpf.jd_short)] < train_limits[0])
                | (tpf.jd_short[np.isfinite(tpf.jd_short)] > train_limits[1]))
    print("Trainging set is {:} out of {:} epochs".format(sum(ok), len(ok)))
    cpm_flux = []
    for t_f in tpf_flux:
        (_, _, signal, time) = cpm_part2.cpm_part2(tpf.jd_short, t_f, None, 
                                tpf_epoch_mask=tpf.epoch_mask,
                                predictor_matrix=predictor_matrix[0],
                                predictor_mask=predictor_mask[0],
                                l2=l2, train_lim=train_limits)
        mask_cpm = tpf.epoch_mask * predictor_mask[0]
        signal_full = np.zeros(len(mask_cpm))
        signal_full[mask_cpm] = signal   
        cpm_flux.append(signal_full)
    cpm_flux = np.array(cpm_flux)

    print("Plot CPM pixel curves:")
    file_name = "example_1_pixel_curves_CPM.png"
    plot_pixel_curves(pixels, cpm_flux[:,mask_cpm], time, file_name)
    print(file_name)
  
    # We need to have vectors of the same shape:
    time_masked = time[mask_prfs[mask_cpm]]
    cpm_flux_masked = cpm_flux[:, mask_cpm * mask_prfs]
    prfs_masked = prfs[mask_cpm * mask_prfs]

    # Final calculation - combine CPM results and PRF information.
    cpm_flux_prfs = cpm_flux.T * prfs
    prfs_square = prfs**2
    prfs_square[~(mask_cpm * mask_prfs)] = 0.
    prfs_square_cumsum = np.cumsum(prfs_square, axis=1)
    # some epochs have to be corrected in order not to have div. by 0.
    prf_sum_limit = 1.e-6
    sel = (prfs_square_cumsum < prf_sum_limit)
    prfs_square_cumsum[sel] = prf_sum_limit
    # And this is the very final calculation:
    result = np.cumsum(cpm_flux_prfs, axis=1) / prfs_square_cumsum
    result = result[mask_cpm * mask_prfs]
    # Also need to mark very large and very small values.
    # I've set these limit after first run, 
    # they will change from object to object.
    lim1 = 499.
    lim2 = -3499.
    sel1 = (result > lim1)
    sel2 = (result < lim2)
    result[sel1] = lim1
    result[sel2] = lim2
    print("values beyond the range: {:} and {:} (out of {:})\n".format(
        np.sum(sel1), np.sum(sel2), result.size))

    print("making plots...\n(different colors mark different number of pixels combined)")
    plot_1_name = "example_1_eb234840_CPM.png"
    plot_2_name = "example_1_eb234840_CPMPRF.png"
    plt.rc('text', usetex=True)
    numbers_to_plot = [0, 1, 2, 9, 14, 19]

    for i in numbers_to_plot:
        plt.plot(time, np.sum(cpm_flux[:i+1, mask_cpm], axis=0), '.', label="{:} pix".format(i+1))
    txt_1 = 'OGLE-BLG-ECL-234840 photometry using CPM ($\lambda = ${:g})'.format(l2)
    finish_figure(plot_1_name, title=txt_1)
    print(plot_1_name)

    for i in numbers_to_plot:
        plt.plot(time[mask_prfs[mask_cpm]], result[:,i], '.', label="{:} pix".format(i+1))
    txt_2 = 'OGLE-BLG-ECL-234840 photometry using CPM+PRF postmortem ($\lambda = ${:g})'.format(l2)
    finish_figure(plot_2_name, title=txt_2)
    print(plot_2_name)

    # In case you want to save smoothed lc 
    if True: # if you want to export model light curve:
    #if False:
        print("\nAdditional plots and data files:")
        half_bin = 12
        lim = -2000.
        file_1 = "example_1_model.dat"
        file_2 = "example_1_model.png"
        file_3 = "example_1_model_averaged.dat"
        file_4 = "example_1_model_averaged.png"
        n_average = 20
        n_average = 19

        raw = np.sum(cpm_flux[:n_average, mask_cpm], axis=0)
        averaged = running_mean_with_edges(raw, half_bin)
        averaged_out = tpf.jd_short * 0.
        averaged_out[tpf.epoch_mask] = np.interp(tpf.jd_short[tpf.epoch_mask], time, averaged)
        indexes = np.arange(len(averaged_out))
        averaged_out[~tpf.epoch_mask] = np.interp(indexes[~tpf.epoch_mask], indexes[tpf.epoch_mask], averaged_out[tpf.epoch_mask])

        np.savetxt(file_1, averaged_out)
        print(file_1)
        plt.plot(tpf.jd_short, averaged_out, '.', label='some')
        finish_figure(file_2, title='averaged {:}'.format(half_bin))
        print(file_2)

        sel = (averaged_out < lim)
        parabola = np.polyfit(np.arange(len(sel))[sel], averaged_out[sel], 2)
        index_minimum = int(-parabola[1] / (2. * parabola[0]))
        print("index of minimum: {:}".format(index_minimum)) # 926
        print("time of minimum: {:}".format(tpf.jd_short[index_minimum])) # 7520.01975

        half_vector = averaged_out[:index_minimum+1]
        add = averaged_out[index_minimum+1:]
        half_vector[-len(add):] += add[::-1]
        half_vector[-len(add):] /= 2.
        output = np.concatenate( (half_vector, half_vector[::-1]) )
        output_time = output * 0.
        output_time[:index_minimum+1] = tpf.jd_short[:index_minimum+1] - tpf.jd_short[index_minimum]
        output_time[index_minimum:-1] = -output_time[:index_minimum+1][::-1]
        output_time[-1] = output_time[-2] + (output_time[-2]-output_time[-3])
        for (i, t) in enumerate(output_time):
            if np.isnan(t):
                output_time[i] = output_time[i-1] + (output_time[i-2]-output_time[i-3])
        np.savetxt(file_3, np.array([output_time, output]).T)
        print(file_3)
        plt.plot(output_time, output, '.', label='some')
        finish_figure(file_4, title='idealized model')
        print(file_4)

