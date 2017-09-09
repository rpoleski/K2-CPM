import numpy as np

from campaigngridradec2pix import CampaignGridRaDec2Pix
from prfdata import PrfData


class PrfForCampaign(object):
    """gives PRF values for a star and set of pixels in given (sub-)campaign"""

    def __init__(self, campaign=None, grids=None, prfdata=None):
        """grids must be of CampaignGridRaDec2Pix type and prfdata must be of PrfData type"""
        if not isinstance(grids, CampaignGridRaDec2Pix):
            raise TypeError("wrong type of of grids argument ({:}), CampaignGridRaDec2Pix expected".format(type(grids)))
        if not isinstance(prfdata, PrfData):
            raise TypeError("wrong type of of prfdata argument ({:}), PrfData expected".format(type(prfdata)))
            
        if campaign is None:
            self.campaign = None
        else:
            self.campaign = int(campaign)

        self.grids = grids
        self.prfdata = prfdata

    def mean_position(self, ra, dec):
        """for given (RA,Dec) calculate position for every epoch
        and report weighted average"""
        return self.grids.mean_position(ra=ra, dec=dec)

    def apply_grids_and_prf(self, ra, dec, pixels, bjds):
        """For star at given (RA,Dec) try to calculate its positions for 
        epochs in bjds and for every epoch predict PRF value for every 
        pixel in pixels (type: list).
        
        Returns:
          failed - number of epochs for which there are no grids
          mask - numpy.array of shape (len(bjds),) that gives the mask for 
            calculated epochs; you will later use bjds[mask] etc. in your code.
          out_prfs - numpy.array of shape (len(bjds), len(pixels)) that gives 
            prf values for all epochs and pixels; remember to use 
            out_prfs[mask]
        """
        if not isinstance(ra, float) or not isinstance(dec, float):
            raise TypeError('wrong types of RA,Dec in apply_grids_and_prf(): {:} and {:}; 2 floats expected'.format(type(ra), type(dec)))

        (positions_x, positions_y) = self.grids.apply_grids(ra=ra, dec=dec)

        n_epochs = len(bjds)
        out_prfs = np.zeros(shape=(n_epochs, len(pixels)))
        mask = np.ones(n_epochs, dtype='bool')
        for i in range(n_epochs):
            try:
                index = self.grids.index_for_bjd(bjds[i])
            except:
                mask[i] = False
                continue
            out_prfs[i] = self.prfdata.get_interpolated_prf(positions_x[index], positions_y[index], pixels)
        
        self._positions_x = positions_x
        self._positions_y = positions_y
        
        failed = len(mask) - sum(mask)
        
        return (failed, mask, out_prfs)

