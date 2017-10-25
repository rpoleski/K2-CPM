# Produces python code with 2 dicts that gives grids for all campaings and channels.

import numpy as np
import sys

from wcsfromtpf import WcsFromTpf
import poly2d
import matrix_xy


def print_80(string):
    """simple function that produces lines shorter than 80 characters for this specific input"""
    if len(string) < 80:
        print(string)
    else:
        n = string[:80].rfind(" ") + 1
        print(string[:n])
        print_80("    " + string[n:])
        
if __name__ == '__main__':
    degree = 2    
    
    for campaign in [91, 92]:
        for channel in [30, 31, 32, 49, 52]:
            wcs = WcsFromTpf(channel=channel, subcampaign=campaign)
    
            (poly_x, poly_y) = poly2d.fit_two_poly_2d(wcs.ra, wcs.dec, 
                                                wcs.pix_x, wcs.pix_y, degree)

            fmt_x = "poly_x[({:}, {:})] = np.array([{:}])"
            coeffs_x = ", ".join([str(x) for x in poly_x])
            print_80(fmt_x.format(campaign, channel, coeffs_x))
            
            fmt_y = "poly_y[({:}, {:})] = np.array([{:}])"
            coeffs_y = ", ".join([str(x) for x in poly_y])
            print_80(fmt_y.format(campaign, channel, coeffs_y))           
    
