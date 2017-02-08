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


def output_ds(out_array, img_params, fn='result.tif'):
    # create output raster data-set
    cols = img_params[0]
    rows = img_params[1]
    bands = 1  # ndvi image needs only one band
    gt = img_params[3]
    proj = img_params[4]
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    out_ras = driver.Create(fn, cols, rows, bands, GDT_Float32)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    out_band = out_ras.GetRasterBand(1)

    out_band.WriteArray(out_array)

    out_band.SetNoDataValue(0)
    out_band.FlushCache()
    out_band.GetStatistics(0, 1)

    return


def compute_ndvi(image):
    ir_band = image.GetRasterBand(4).ReadAsArray(0, 0).astype(np.float16)
    nir_band = image.GetRasterBand(5).ReadAsArray(0, 0).astype(np.float16)

    mask = np.greater(ir_band + nir_band, 0)
    ndvi = np.choose(mask, (0, (nir_band - ir_band) / (nir_band + ir_band)))
    #ndvi = (nir_band - ir_band) / (nir_band + ir_band + 0.00000000001)
    #ndvi = nir_band - ir_band

    return ndvi


def downscale_image():
    # compute downscale factor by using image parameters of landsat and wv2
    pass


def temporal_mask():
    pass


def main():
    # Open Landsat and WV2 ndvi images
    #landsat_dir = r"subject image/sub.vrt"
    landsat_dir = r"regression_results.tif"
    wv2_dir = r""

    landsat_img = open_image(landsat_dir)

    # retrieve image parameters
    landsat_param = get_img_param(landsat_img)
    #print landsat_param

    # compute landsat ndvi
    ndvi = compute_ndvi(landsat_img)

    output_ds(ndvi, landsat_param, fn='ndvi.tif')

if __name__ == "__main__":
    start = tm.time()
    main()
    print 'Processing time: %f seconds' % (tm.time() - start)
