import gdal
import sys
import time as tm
import pandas as pd
import scipy.stats as st
import numpy as np
import numpy.ma as ma
import os
from gdalconst import *
from subprocess import call

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


def rasterize(src, img_param, fn='result.tif'):
    """
    Converts a region of interest vector data-set in .SHP file to a raster geotiff.
    Computes the extents from the geotransform of the source image of the roi.
    """

    # collect extent, resolution and geotrans, and proj of img to be masked
    topleft_x = img_param[3][0]
    topleft_y = img_param[3][3]
    x = img_param[3][1]
    y = img_param[3][5]

    # compute extents
    x_min = topleft_x
    y_min = topleft_y + y*img_param[1]
    x_max = topleft_x + x*img_param[0]
    y_max = topleft_y

    # gdal command construction from variables
    rasterize_cmd = ['gdal_rasterize',
                     '-a','Id',
                     # '-a_nodata', '0',
                     '-ot', 'UInt16',
                     '-te',  # specify extent
                     str(x_min), str(y_min),
                     str(x_max), str(y_max),
                     '-ts',  # specify the number of columns and rows
                     str(img_param[0]), str(img_param[1]),
                     '-l', os.path.splitext(os.path.basename(src))[0],  # layer name
                     src, fn]

    call(rasterize_cmd)

    return


def create_objects(image, grid):
    """
    http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
    reference for pairing the training segment with the image pixels

    https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb
    reference for creating objects

    The image array and grid array must have the same shape.
    """

    img = image.GetRasterBand(1)
    img_novalue = img.GetNoDataValue()
    img_arr = img.ReadAsArray(0, 0)
    grid = grid.GetRasterBand(1)
    grid_arr = grid.ReadAsArray(0, 0)

    print 'image array has shape {}'.format(img_arr.shape)
    print 'grid array has shape {}'.format(grid_arr.shape)
    print 'image array nodata value: {}'.format(img_novalue)
    # obj_id = pd.Series(np.unique(grid_arr))
    # print img_arr[grid_arr==1851]

    img_arr[img_arr>1] = np.nan
    # mask = img_arr > 1
    # masked_arr = ma.array(img_arr, mask=mask)
    # print masked_arr
    cell_id = []
    cell = []

    # compute statistics for each object
    for i in np.unique(grid_arr):
        obj = img_arr[grid_arr==i]
        # print st.describe(obj, nan_policy='omit')
        no_nan = obj[~np.isnan(obj)]
        if len(no_nan) >= 9:
            result = st.describe(no_nan, nan_policy='omit')
            cell_id.append(i)
            stats = list(result.minmax) + list(result)[2:]
            cell.append(stats)

    objects = pd.DataFrame(cell, index=cell_id,
                           columns=['min', 'max', 'mean',
                                    'variance', 'skewness',
                                    'kurtosis'])

    return objects


def main():
    img_dir = "resampled.tif"
    training_dir = ""
    poly_grid_dir = "landuse_grid100.shp"
    grid_dir = "grid100.tif"

    img = open_image(img_dir)
    img_param = get_img_param(img)

    lu_grid = open_image(grid_dir)
    lu_grid_param = get_img_param(lu_grid)

    # rasterize(poly_grid_dir, img_param)

    obj = create_objects(img, lu_grid)
    #
    # tr_sites = open_image(training_dir)


if __name__ == "__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)