import sys
import gdal
import os
import glob
import time as tm
import numpy as np
from gdalconst import *
from subprocess import call


def get_img_param(image_dataset):
    cols = image_dataset.RasterXSize
    rows = image_dataset.RasterYSize
    num_bands = image_dataset.RasterCount
    img_gt = image_dataset.GetGeoTransform()
    img_proj = image_dataset.GetProjection()
    img_driver = image_dataset.GetDriver()

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    return img_params


def rasterize_mask(src, img_param):
    """
    Converts polygons in .SHP file format into a raster geotiff.
    Uses parameters of the source image that the cloud formations were derived from.
    """
    # collect columns, rows, extent, resolution and geotrans, and proj of img to be masked
    cols = img_param[0]
    rows = img_param[1]
    geotrans = img_param[3]

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


def mask_image(band_list, mask_band, img_params, out_fn='result.tif'):

    # create output raster data-set
    cols = img_params[0]
    rows = img_params[1]
    bands = img_params[2]
    gt = img_params[3]
    proj = img_params[4]
    driver = img_params[5]

    out_ras = driver.Create(out_fn, cols, rows, bands, GDT_UInt32, options=[])
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    for band in band_list:

        no_value = band_list[band].GetNoDataValue()
        print 'band %d pixel no value: %d' % (band, no_value)

        out_band = out_ras.GetRasterBand(band)

        x_bsize = 5000
        y_bsize = 5000
        print '\n masking band %d...' % band
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
                # print band_ds

                # mask the data-set
                # band_ds[band_ds == no_value] = 15  # set the no-value pixels to 0

                mask_array = mask_band.ReadAsArray(j, i, num_cols, num_rows).\
                    astype(np.uint16)

                clear_pixels = np.where(mask_array == 0, np.array(no_value), band_ds)

                out_band.WriteArray(clear_pixels, j, i)

                band_ds = None
                mask_array = None
                clear_pixels = None


        out_band.SetNoDataValue(no_value)
        out_band.FlushCache()
        out_band.GetStatistics(0, 1)

    return


def compress_image(fn, out_fn='compressed.tif'):
    """
    Compresses a geotiff image.
    Referenced from http://www.digital-geography.com/geotiff-compression-comparison/
    and
    http://gis.stackexchange.com/questions/1104/
    should-gdal-be-set-to-produce-geotiff-files-with-compression-which-algorithm-sh
    """
    compress_cmd = ['gdal_translate',
                     '-of', 'GTiff',
                     '-co', 'COMPRESS=LZW',  # use LZW compression algorithm
                     '-co', 'PREDICTOR=2',
                     '-co', 'TILED=YES',
                     fn, out_fn  # layer name
                    ]

    call(compress_cmd)

def main():
    start = tm.time()

    # Worldview2
    img_fn = "landcover_3_clip_reclass_clip.tif"
    poly_fn = "urban_mask.shp"

    # Landsat
    #img_fn = "G:\LUIGI\ICLEI\IMAGE PROCESSING\IMAGES\LC81140512014344LGN00\CLIP\LC81140512014344LGN00_BANDSTACK"
    #poly_fn = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\1a. IMAGE PREPROCESSING\WORKING FILES\\nodata.shp"

    # open image
    img = gdal.Open(img_fn, GA_ReadOnly)

    if img is None:
        print 'Could not open ' + img_fn
        sys.exit(1)

    # TODO: automate data type assignment for output data-set

    wv2_param = get_img_param(img)

    # rasterize_mask(poly_fn, wv2_param)

    # collect all bands
    b_list = {}
    for band in range(wv2_param[2]):
        b_list[band+1] = img.GetRasterBand(band+1)

    # search for output mask from the previous step
    cwd = os.getcwd()
    for f in glob.glob(cwd + '\*_mask.tif'):  # search for the .tif file of the mask
        mask_img = gdal.Open(f, GA_ReadOnly)
        band_mask = mask_img.GetRasterBand(1)
        mask_image(b_list, band_mask, wv2_param, 'landsat_urban_masked.tif')

    # compress image
    for f in glob.glob(cwd + '\*_masked.tif'):
        compress_image(f)

    print 'Processing time: %f' % (tm.time() - start)


if __name__ == "__main__":
    main()
