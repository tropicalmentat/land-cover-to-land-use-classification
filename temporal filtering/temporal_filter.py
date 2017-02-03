#compute Landsat NDVI
#compute WV2 NDVI
#downscale WV2 NDVI to Landsat resolution
#perform linear regression

import gdal
import time as tm
import numpy as np
import scipy
import scipy.stats
from gdalconst import *
from skimage import exposure
from skimage import io
from sklearn import tree
from matplotlib import pyplot as plt
from matplotlib import colors


def compute_ndvi(band_array):
    pass


def downscale_image():
    # compute downscale factor
    pass


def temporal_mask():
    pass


def main():
    # Open Landsat
    landsat_dir = r""
    wv2_dir = r""

    # compute landsat ndvi


if __name__ == "__main__":
    start = tm.time()
    main()
    print 'Processing time: %f seconds' % (tm.time() - start)
