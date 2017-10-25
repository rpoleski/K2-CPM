# A script for changing format of WcsFromTpf files so that they can 
# be nicely transferred to a new repo (MCPM).

import numpy as np

from wcsfromtpf import WcsFromTpf
import poly2d
import matrix_xy


def get_pixel_properties(wcs):
    """parse WcsFromTpf into 2 dicts that have all pixels for each epic"""
    pixels_x = {}
    pixels_y = {}
    for i in range(len(wcs.epic)):
        epic = wcs.epic[i]
        if epic not in pixels_x:
            pixels_x[epic] = []
            pixels_y[epic] = []
        pixels_x[epic].append(wcs.pix_x[i])
        pixels_y[epic].append(wcs.pix_y[i])
    return (pixels_x, pixels_y)
    
def get_ref_range(pixels):
    """get minimum, maximum, and full range (n+1) of values in the list"""
    min_ = min(pixels)
    max_ = max(pixels)
    return (min_, max_, max_-min_+1)
    
def read_weird_cases(file_name):
    """read a file that has manually prepared info on strange cases"""
    out = {}
    with open(file_name) as data:
        for line in data.readlines():
            columns = line.split()
            if columns[0] not in out:
                out[columns[0]] = []
            out[columns[0]].append(np.array([int(x) for x in columns[1:]]))
    return out

if __name__ == '__main__':
    file_name = 'weird_rectangles.dat'
    
    weird = read_weird_cases(file_name)
        
    for campaign in [91, 92]:
        for channel in [30, 31, 32, 49, 52]:
            
            wcs = WcsFromTpf(channel=channel, subcampaign=campaign)
            
            (pix_x, pix_y) = get_pixel_properties(wcs)
            
            out = []
            for epic in pix_x:
                n_pix = len(pix_x[epic])
                (min_x, max_x, d_x) = get_ref_range(pix_x[epic])
                (min_y, max_y, d_y) = get_ref_range(pix_y[epic])
                if n_pix > 0.9 * d_x * d_y: # These are full and almost-full rectangles.
                    out.append([epic, min_x, max_x, min_y, max_y])
                    continue
                if epic in weird: # These were manually vetted.
                    for rectangle in weird[epic]:
                        x_a = min_x + rectangle[0]
                        x_b = min_x + rectangle[1]
                        y_a = min_y + rectangle[2]
                        y_b = min_y + rectangle[3]
                        out.append([epic, x_a, x_b, y_a, y_b])
                    continue
                raise ValueError('This should never happen')
            
            file_out = "../../data_K2C9/tpf_rectangles_{:}_{:}.data".format(campaign, channel)
            with open(file_out, 'w') as output:
                for line in out:
                    output.write("{:} {:4} {:4} {:4} {:4}\n".format(*line))
            print(file_out + " is done")
            