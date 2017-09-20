import sys
import numpy as np
import matplotlib.pyplot as plt

from prfdata import PrfData
from campaigngridradec2pix import CampaignGridRaDec2Pix
from prfforcampaign import PrfForCampaign
import cpm_part1
import cpm_part2
import tpfdata


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
    campaign_train = 92
    ra = 269.929125
    dec = -28.410833

    # Now we need to know the average position of the target. For this we 
    # need to setup grid transformations:
    grid_name = '../../data_K2C9/grids_RADEC2pix_91_{:}.data'.format(channel)
    grids = CampaignGridRaDec2Pix(campaign=campaign, file_name=grid_name)
    # and then get mean position:
    (mean_x, mean_y) = grids.mean_position(ra, dec)
    print("Mean target position: {:.2f} {:.2f}\n".format(mean_x, mean_y))

    # Now we want to select a few pixels with most flux. First, let's make 
    # temporary list of pixels, let's say 5x5 i.e., "half_size" of 2:
    half_size = 2
    int_x = int(mean_x+0.5)
    int_y = int(mean_y+0.5)
    pixels = np.mgrid[(int_x-half_size):(int_x+half_size+1),
                    (int_y-half_size):(int_y+half_size+1)].reshape(2, -1).T
    
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
    prfs_bjds = prf_for_campaign.grids.bjd_array - 2450000.

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
    n_predictor = 1000
    n_pca = 0
    distance = 16
    exclusion = 1
    flux_lim = (0.2, 1.5)
    tpf_dir = "tpf/" # In particular, the path to tpf data has to be given.
    # We in fact run it only for a single central pixel (see 
    # pixel_list=np.array([pixels[0]]) ), it won't affect results and 
    # makes life easier.
#    (predictor_matrix, predictor_mask) = cpm_part1.run_cpm_part1(
#            channel=channel, campaign=campaign, num_predictor=n_predictor,
#            num_pca=n_pca, dis=distance, excl=exclusion, flux_lim=flux_lim, 
#            input_dir=tpf_dir, pixel_list=np.array([pixels[0]]), 
#            train_lim=None, return_predictor_epoch_masks=True)
    (predictor_matrix, predictor_matrix_apply, predictor_mask, predictor_mask_apply) = cpm_part1.run_cpm_part1(
            channel=channel, campaign=campaign_train, num_predictor=n_predictor,
            num_pca=n_pca, dis=distance, excl=exclusion, flux_lim=flux_lim, 
            input_dir=tpf_dir, pixel_list=np.array([pixels[0]]), 
            train_lim=None, return_predictor_epoch_masks=True, 
            campaign_2=campaign)
            
    print("Done!\n")

    # Now extract raw light ruves from TPF files:
    epic_number = '200069761' # in case you don't know it, run:
    # >>> from K2CPM import wcsfromtpf
    # >>> epic_number = wcsfromtpf.WcsFromTpf(channel, campaign).get_nearest_pixel_radec(ra, dec)[4]
    tpfdata.TpfData.directory = tpf_dir
    #tpf = tpfdata.TpfData(epic_id=epic_number, campaign=campaign)
    tpf_train = tpfdata.TpfData(epic_id=epic_number, campaign=campaign_train)
    tpf_apply = tpfdata.TpfData(epic_id=epic_number, campaign=campaign)
    tpf_flux_train = []
    tpf_flux_apply = []
    for (i, pixel) in enumerate(pixels):
        tpf_flux_train.append(tpf_train.get_flux_for_pixel(row=pixel[1], column=pixel[0]))
        tpf_flux_apply.append(tpf_apply.get_flux_for_pixel(row=pixel[1], column=pixel[0]))

    # Now run cpm_part_2:
    l2 = 1.e3 # This is regularization strength.
    #train_limits = [7508., 7540.] # We train on data before 2457508.
    train_limits = [7528.5, 7532.5]
    ok = ((tpf_train.jd_short[np.isfinite(tpf_train.jd_short)] < train_limits[0])
        | (tpf_train.jd_short[np.isfinite(tpf_train.jd_short)] > train_limits[1]))
    print("Trainging set is {:} out of {:} epochs".format(sum(ok), len(ok)))
    cpm_flux = []
    for i in range(len(pixels)):
        (_, _, signal, time) = cpm_part2.cpm_part2_subcampaigns(
                tpf_train.jd_short, tpf_flux_train[i], None, 
                tpf_train.epoch_mask, predictor_matrix[0], predictor_mask[0], 
                tpf_apply.jd_short, tpf_flux_apply[i], None,
                tpf_apply.epoch_mask, predictor_matrix_apply[0], predictor_mask_apply[0],
                l2=l2, train_lim=train_limits)
        cpm_flux.append(signal)
    cpm_flux = np.array(cpm_flux)
  
    # We need to have the same time vector - that's what we do here.
    # Note that some epochs lack astrometric solutions.
    index = []
    mask = np.ones(len(time), dtype='bool')
    for (i, t) in enumerate(time):
        try:
            index.append(prf_for_campaign.grids.index_for_bjd(t+2450000.))
        except:
            mask[i] = False
    time_masked = time[mask]
    cpm_flux_masked = cpm_flux[:, mask]
    prfs_masked = prfs[index]

    # Final calculation - combine CPM results and PRF information.
    cpmf_flux_prfs_masked = cpm_flux_masked.T * prfs_masked
    prfs_square_masked = prfs_masked**2
    prfs_square_masked_cumsum = np.cumsum(prfs_square_masked, axis=1)
    # some epochs have to be corrected in order not to have div. by 0.
    prf_sum_limit = 1.e-6
    sel = (prfs_square_masked_cumsum < prf_sum_limit)
    prfs_square_masked_cumsum[sel] = prf_sum_limit
    # And this is the very final calculation:
    result = np.cumsum(cpmf_flux_prfs_masked, axis=1) / prfs_square_masked_cumsum 
    # Also need to mark very large and very small values.
    # I've set these limit after first run, 
    # they will change from object to object.
    lim1 = 1999.
    lim2 = -5499.
    sel1 = (result > lim1)
    sel2 = (result < lim2)
    result[sel1] = lim1
    result[sel2] = lim2
    print("values beyond the range: {:} and {:} (out of {:})\n".format(
        np.sum(sel1), np.sum(sel2), result.size))

    print("making plots...\n(different colors mark different number of pixels combined)")
    plot_1_name = "example_2_eb234840_CPM.png"
    plot_2_name = "example_2_eb234840_CPMPRF.png"
    plt.rc('text', usetex=True)
    numbers_to_plot = [0, 1, 2, 9, 14, 19]

    for i in numbers_to_plot:
        out = np.sum(cpm_flux[:i+1,:], axis=0)
        sel1 = (out > lim1)
        sel2 = (out < lim2)
        out[sel1] = lim1
        out[sel2] = lim2
        plt.plot(time, out, '.', label="{:} pix".format(i+1))
    txt_1 = 'OGLE-BLG-ECL-234840 photometry using CPM(trained on 92) ($\lambda = ${:g})'.format(l2)
    finish_figure(plot_1_name, title=txt_1)
    print(plot_1_name)

    for i in numbers_to_plot:
        plt.plot(time_masked, result[:,i], '.', label="{:} pix".format(i+1))
    txt_2 = 'OGLE-BLG-ECL-234840 photometry using CPM(trained on 92)+PRF  ($\lambda = ${:g})'.format(l2)
    finish_figure(plot_2_name, title=txt_2)
    print(plot_2_name)

