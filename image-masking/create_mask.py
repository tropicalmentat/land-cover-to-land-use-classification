import sys
import gdal
import os
import glob
import time as tm
import numpy as np
from gdalconst import *
from subprocess import call


def output_ds(out_array, img_params, fn='result.tif'):
    # create output raster data-set
    cols = img_params[0]
    rows = img_params[1]
    bands = img_params[2]
    gt = img_params[3]
    proj = img_params[4]
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    out_ras = driver.Create(fn, cols, rows, bands, GDT_UInt16)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    for band in range(out_array.shape[2]):
        out_band = out_ras.GetRasterBand(band+1)

        out_band.WriteArray(out_array[:, :, band])

        out_band.SetNoDataValue(0)
        out_band.FlushCache()
        out_band.GetStatistics(0, 1)

    return


def rasterize_mask(src, geotrans, cols, rows):
    """
    Converts polygons in .SHP file format into a raster geotiff.
    Uses parameters of the source image that the cloud formations were derived from.
    :param src: directory of source image
    :param geotrans: source image geotransform
    :param cols: source image number of columns
    :param rows: source image number of rows
    :return:
    """
    # collect extent, resolution and geotrans, and proj of img to be masked
    topleft_x = geotrans[0]
    topleft_y = geotrans[3]
    x = geotrans[1]
    y = geotrans[5]

    #wgs84 = Proj(proj='latlong', ellps='WGS84')
    #utm51n = Proj(proj='utm', zone=51, ellps='WGS84')

    # compute extents
    x_min = topleft_x
    y_min = topleft_y + y*rows
    x_max = topleft_x + x*cols
    y_max = topleft_y

    out_fn = os.path.splitext(os.path.basename(src))[0] + '.tif'

    # gdal command construction from variables
    rasterize_cmd = ['gdal_rasterize',
                     '-i',  # inverse rasterization
                     '-ot', 'UInt16',
                     '-te',  # specify extent
                     str(x_min), str(y_min),
                     str(x_max), str(y_max),
                     '-ts',  # specify the number of columns and rows
                     str(cols), str(rows),
                     '-burn', '1',  # value of cloud pixels for mask building
                     '-l', os.path.splitext(os.path.basename(src))[0],  # layer name
                     src, out_fn]

    call(rasterize_cmd)

    return


def mask_image(band_list, mask_band, img_params):

    # create output raster data-set
    cols = img_params[0]
    rows = img_params[1]
    bands = img_params[2]
    gt = img_params[3]
    proj = img_params[4]
    driver = img_params[5]

    out_ras = driver.Create('naga_urban_masked.tif', cols, rows, bands, GDT_UInt16)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    for band in band_list:
        print '\n masking band %d...' % band
        no_value = band_list[band].GetNoDataValue()
        out_band = out_ras.GetRasterBand(band)

        x_bsize = 5000
        y_bsize = 5000

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


                band_ds = band_list[band].ReadAsArray(j, i, num_cols, num_rows).\
                    astype(np.uint16)

                # mask the data-set
                novalue_mask = np.where(band_ds == no_value, 0, band_ds) # set the no-value pixels to 0

                mask_array = mask_band.ReadAsArray(j, i, num_cols, num_rows).\
                    astype(np.uint16)

                clear_pixels = novalue_mask * mask_array
                #cp_shape = clear_pixels.shape
                #print clear_pixels

                out_band.WriteArray(clear_pixels, j, i)


        out_band.SetNoDataValue(0)
        out_band.FlushCache()
        out_band.GetStatistics(0, 1)

    #print mask_ds.shape
    return


def main():
    start = tm.time()

    # Worldview2
    img_fn = "naga_urb.tif"
    poly_fn = "river_cloud_mask.shp"

    # Landsat
    #img_fn = "G:\LUIGI\ICLEI\IMAGE PROCESSING\IMAGES\LC81140512014344LGN00\CLIP\LC81140512014344LGN00_BANDSTACK"
    #poly_fn = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\1a. IMAGE PREPROCESSING\WORKING FILES\\nodata.shp"

    # open image
    img = gdal.Open(img_fn, GA_ReadOnly)

    if img is None:
        print 'Could not open ' + img_fn
        sys.exit(1)

    # TODO: automate data type assignment for output data-set
    cols = img.RasterXSize
    rows = img.RasterYSize
    num_bands = img.RasterCount
    img_gt = img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_driver = img.GetDriver()

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    #rasterize_mask(poly_fn, img_gt, cols, rows)

    # collect all bands
    b_list = {}
    for band in range(num_bands):
        #pass
        b_list[band+1] = img.GetRasterBand(band+1)

    # search for output mask from the previous step
    cwd = os.getcwd()
    for f in glob.glob(cwd + '\*_mask.tif'):  # search for the .tif file of the mask
        mask_img = gdal.Open(f, GA_ReadOnly)
        band_mask = mask_img.GetRasterBand(1)
        mask_image(b_list, band_mask, img_params)

    print 'Processing time: %f' % (tm.time() - start)


if __name__ == "__main__":
    main()
