# references
# http://chris35wills.github.io/courses/pydata_stack/

from subprocess import call
import sys
import gdal
from gdalconst import *
# from skimage.transform import downscale_local_mean
# from scipy.ndimage import median_filter
import pandas as pd
import numpy as np
# import xarray as xr
import time as tm
# from skimage.measure import block_reduce


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


def output_ds(out_array, img_params, d_type=GDT_Unknown, fn='result.tif'):
    """
    Helper function.
    Writes new data-set into disk
    and saves output arrays in the data-set.
    """

    # create output raster data-set
    cols = img_params[0]
    rows = img_params[1]
    bands = 1  # ndvi image needs only one band
    gt = img_params[3]
    proj = img_params[4]
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    out_ras = driver.Create(fn, cols, rows, bands, d_type)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    out_band = out_ras.GetRasterBand(1)

    out_band.WriteArray(out_array)

    out_band.SetNoDataValue(-99)
    out_band.FlushCache()
    out_band.GetStatistics(0, 1)

    return


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
                      '-ot', 'Float32',
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


def map_impervious(ref_img, ndvi, img1_param, img2_param, fn='impervious_surfaces.tif'):

    col_ds_factor = img1_param[0]/float(img2_param[0])
    print int(col_ds_factor)
    row_ds_factor = img1_param[1]/float(img2_param[1])
    print int(row_ds_factor)

    ref_var = ref_img.GetRasterBand(1)
    ref_var_no_value = ref_var.GetNoDataValue()
    ref_arr = ref_var.ReadAsArray(0, 0).astype(np.float16)
    ref_arr[ref_arr==ref_var_no_value] = np.nan

    # br = block_reduce(ref_arr, (55, 55), func=np.sum)

    # a solution for downscaling the worldview2 image can be found here
    # http://stackoverflow.com/questions/890128/why-are-python-lambdas-useful
    df = pd.DataFrame(ref_arr)
    downscaled = df.groupby(lambda x: int(x/col_ds_factor)).mean().\
        groupby(lambda y: int(y/row_ds_factor), axis=1).mean()
    downscaled1 = downscaled.drop(197, 0)
    print downscaled1
    ods = np.array(downscaled1)
    print ods[~np.isnan(ods)]
    output_ds(ods, img2_param, GDT_Float32)




    return


def main():
    img_dir = 'vegetation-impervious_postprocessed.tif'
    landsat_dir = 'landsat_ndvi_masked.tif'

    ndvi_img = open_image(landsat_dir)
    ndvi_param = get_img_param(ndvi_img)
    print 'The landsat NDVI image has \n{} columns \n{} rows'\
        .format(ndvi_param[0], ndvi_param[1])

    img = open_image(img_dir)
    img_param = get_img_param(img)
    print '\nThe worldview2 image has \n{} columns\n{} rows'\
        .format(img_param[0], img_param[1])

    print '\n{} {}'.\
        format(img_param[0]/float(ndvi_param[0]),img_param[1]/float(ndvi_param[1]))

    # map_impervious(img, ndvi_img, img_param, ndvi_param)

    downscale_image(img_dir, ndvi_param)

    # resampled_ = open_image('resampled.tif')
    # resampled_param = get_img_param(resampled_)

if __name__=="__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)