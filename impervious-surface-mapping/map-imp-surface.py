from subprocess import call
import sys
import gdal
from gdalconst import *
from skimage.transform import downscale_local_mean
from scipy.ndimage import median_filter
import pandas as pd
import numpy as np


def open_image(directory):
    """
    Helper function.
    Opens image and returns
    gdal MajorObject
    """
    image_ds = gdal.Open(directory, GA_ReadOnly)

    if image_ds is None:
        print 'Could not open ' + directory
        sys.exit(1)

    return image_ds


def get_img_param(image_dataset):
    """
    Helper function.
    Collects image parameters
    returns them as a list.
    """
    cols = image_dataset.RasterXSize
    rows = image_dataset.RasterYSize
    num_bands = image_dataset.RasterCount
    img_gt = image_dataset.GetGeoTransform()
    img_proj = image_dataset.GetProjection()
    img_driver = image_dataset.GetDriver()

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    return img_params


def downscale_image(hires_img, lores_param, fn='resampled.tif'):
    """
    Takes in a high resolution image and resamples it
    to a lower resolution.
    The parameters are taken from a low resolution
    reference image that are used to compute the extent
    and resolution of the resampled data-set.
    """

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

    downsample_cmd = [
                      'gdalwarp',
                      '-r', 'average',  # use average as sampling algorithm
                      '-te',  # specify extent
                      str(x_min), str(y_min),
                      str(x_max), str(y_max),
                      '-ts',  # specify the number of columns and rows
                      str(cols), str(rows),
                      hires_img,
                      fn
                      ]

    call(downsample_cmd)

    return


def map_impervious(ref_img, ndvi, fn='impervious_surfaces.tif'):

    ref_var = ref_img.GetRasterBand(1)
    no_value = ref_var.GetNoDataValue()
    ref_arr = ref_var.ReadAsArray(0, 0)
    # a = ref_arr[ref_arr==no_value] = 0

    # print ref_var

    # downscaled = downscale_local_mean(ref_var, (290, 197))
    # print downscaled
    # median = median_filter(ref_var, (55,55))
    # print median[median!=15]
    df = pd.DataFrame(ref_arr)
    print df

    return


def main():
    img_dir = 'vegetation-impervious_postprocessed.tif'
    landsat_dir = 'landsat_ndvi_masked.tif'

    ndvi_img = open_image(landsat_dir)
    ndvi_param = get_img_param(ndvi_img)
    print ndvi_param[0], ndvi_param[1]

    img = open_image(img_dir)
    img_param = get_img_param(img)
    print img_param[0], img_param[1]

    print img_param[0]/ndvi_param[0],img_param[1]/ndvi_param[1]

    map_impervious(img, ndvi_img)

    # downscale_image(img_dir, ndvi_param)

    # resampled_ = open_image('resampled.tif')
    # resampled_param = get_img_param(resampled_)


    # map_impervious(resa)

if __name__=="__main__":
    main()