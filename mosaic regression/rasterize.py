import os
import sys
import time as tm
import gdal
from gdalconst import *
from subprocess import call


# rasterize cloud and shadow polygon
def rasterize_clouds(src, geotrans, cols, rows):
    """
    Converts cloud formations in vector .SHP file format into a raster geotiff.
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


def main():
    start = tm.time()

    # Landsat
    submask_dir = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\thesis-project-scripts\mosaic regression\sub_mask.shp"
    refmask_dir = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\thesis-project-scripts\mosaic regression\\ref_mask.shp"
    union_mask = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\thesis-project-scripts\mosaic regression\\union_mask.shp"
    img_dir = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\thesis-project-scripts\mosaic regression\\ref.vrt"

    # open image
    img = gdal.Open(img_dir, GA_ReadOnly)

    if img is None:
        print 'Could not open ' + img_dir
        sys.exit(1)

    # TODO: automate data type assignment for output data-set
    cols = img.RasterXSize
    rows = img.RasterYSize
    num_bands = img.RasterCount
    img_gt = img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_driver = img.GetDriver()

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    # rasterize clouds and shadows for subject and reference image and the union of both masks
    rasterize_clouds(submask_dir, img_gt, cols, rows)
    rasterize_clouds(refmask_dir, img_gt, cols, rows)
    rasterize_clouds(union_mask, img_gt, cols, rows)

    print 'Processing time: %f' % (tm.time() - start)


if __name__ == "__main__":
    main()