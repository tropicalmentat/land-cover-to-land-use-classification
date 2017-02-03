#compute Landsat NDVI
#compute WV2 NDVI
#downscale WV2 NDVI to Landsat resolution
#perform linear regression

import gdal
import sys
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


def open_image(directory):
    image_ds = gdal.Open(directory, GA_ReadOnly)

    if image_ds is None:
        print 'Could not open ' + directory
        sys.exit(1)

    return image_ds


def get_img_param(image_dataset):
    cols = image_dataset.RasterXSize
    rows = image_dataset.RasterYSize
    num_bands = image_dataset.RasterCount
    img_gt = image_dataset.GetGeoTransform()
    img_proj = image_dataset.GetProjection()
    img_driver = image_dataset.GetDriver()

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    return img_params


def compute_ndvi(band_array):
    pass


def downscale_image():
    # compute downscale factor by using image parameters of landsat and wv2
    pass


def temporal_mask():
    pass


def main():
    # Open Landsat
    landsat_dir = r""
    wv2_dir = r""

    # retrieve image parameters

    # compute landsat ndvi


if __name__ == "__main__":
    start = tm.time()
    main()
    print 'Processing time: %f seconds' % (tm.time() - start)
