import gdal
import sys
import time as tm
import numpy as np
from gdalconst import *


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


def compute_ndvi(image, img_params, ir=3, nir=4, fn='ndvi.tif'):
    """
    Computes Normalized Difference Vegetation Index (NDVI)
    of multispectral images with Infrared (IR) and Near Infrared (NIR) bands.
    Returns array with ndvi values with shape of original data.
    -------------------------------------------------------------------------
    Creation of new gdal data-set is included in this function
    to avoid the MemoryError that numpy throws when a separate large
    array is created (eg. Worldview2 Pansharpened Image)
    """

    print '\ncomputing ndvi...'
    # create output raster data-set
    cols = img_params[0]
    rows = img_params[1]
    bands = 1  # the ndvi output data-set needs only 1 band
    gt = img_params[3]
    proj = img_params[4]
    driver = img_params[5]

    out_ras = driver.Create(fn, cols, rows, bands, GDT_Float32)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    out_band = out_ras.GetRasterBand(1)

    # compute ndvi per block
    x_bsize = 5000
    y_bsize = 5000

    ir_band = image.GetRasterBand(ir)
    nir_band = image.GetRasterBand(nir)

    for i in range(0, rows, y_bsize):
        if i + y_bsize < rows:
            num_rows = y_bsize
        else:
            num_rows = rows - i
        for j in range(0, cols, x_bsize):
            if j + x_bsize < cols:
                num_cols = x_bsize
            else:
                num_cols = cols - j

            ir_array = ir_band.ReadAsArray(j, i, num_cols, num_rows).\
                astype(np.float16)
            nir_array = nir_band.ReadAsArray(j, i, num_cols, num_rows).\
                astype(np.float16)

            mask = np.greater(ir_array + nir_array, 0)
            ndvi = np.choose(mask, (-99, (nir_array - ir_array) / (nir_array + ir_array)))

            out_band.WriteArray(ndvi, j, i)

    out_band.SetNoDataValue(-99)
    out_band.FlushCache()
    out_band.GetStatistics(0, 1)

    return


def main():
    # open Worldview2 pansharpened image
    wv2_dir = "naga_urban_masked.tif"
    wv2_img = open_image(wv2_dir)

    # retrieve Worldview2 image parameters
    wv2_param = get_img_param(wv2_img)

    # compute Worldview2 ndvi
    compute_ndvi(wv2_img, wv2_param, ir=3, nir=4, fn='wv2_ndvi.tif')

    # open Landsat multispectral image
    landsat_dir = "landsat_urban.tif"
    landsat_img = open_image(landsat_dir)

    # retrieve Landsat image parameters
    landsat_param = get_img_param(landsat_img)

    # compute Landsat ndvi
    compute_ndvi(landsat_img, landsat_param, ir=4, nir=5, fn='landsat_ndvi.tif')


if __name__ == "__main__":
    start = tm.time()
    main()
    print 'Processing time: %f seconds' % (tm.time() - start)
