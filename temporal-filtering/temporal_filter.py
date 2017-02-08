import gdal
import sys
import os
import time as tm
import numpy as np
import scipy
import scipy.ndimage
import scipy.stats
from gdalconst import *
from subprocess import call


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


def downscale_image(hires_img, lores_param):
    # compute downscale factor by using image parameters of landsat and wv2

    # collect columns, rows, extent, resolution and geotrans, and proj of img to be masked
    cols = lores_param[0]
    rows = lores_param[1]
    geotrans = lores_param[3]

    # unpack geotransform parameters
    topleft_x = geotrans[0]
    topleft_y = geotrans[3]
    x = geotrans[1]
    y = geotrans[5]

    # compute extents
    x_min = topleft_x
    y_min = topleft_y + y*rows
    x_max = topleft_x + x*cols
    y_max = topleft_y

    downsample_cmd = ['gdalwarp',
                      '-r', 'average',
                      '-te',  # specify extent
                      str(x_min), str(y_min),
                      str(x_max), str(y_max),
                      '-ts',  # specify the number of columns and rows
                      str(cols), str(rows),
                      hires_img,
                      'wv2_ndvi_resampled.tif'
                      ]

    call(downsample_cmd)

    return


def temporal_mask():
    pass


def main():
    # Open Landsat and WV2 ndvi images
    landsat_dir = "landsat_ndvi.tif"
    wv2_dir = "wv2_ndvi.tif"

    landsat_img = open_image(landsat_dir)
    wv2_img = open_image(wv2_dir)

    # retrieve image parameters
    landsat_param = get_img_param(landsat_img)
    print 'Landsat8 image has: \n%d columns\n%d rows' % (landsat_param[0], landsat_param[1])
    wv2_param = get_img_param(wv2_img)
    print '\nWorldview2 image has: \n%d columns\n%d rows' % (wv2_param[0], wv2_param[1])

    # downscale wv2 image
    cwd = os.getcwd()
    print cwd
    downscale_image(wv2_dir, landsat_param)

if __name__ == "__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)
