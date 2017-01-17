import sys
import gdal
import os
import sys
import time as tm
import numpy as np
import scipy
import scipy.stats
from gdalconst import *
from skimage import exposure
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from matplotlib import colors
from subprocess import call

io.use_plugin('matplotlib')
# to get matplotlib display images the following was taken from
# https://github.com/scikit-image/scikit-image/issues/599


def rasterize_roi(src, geotrans, cols, rows):
    """
    Converts a region of interest vector data-set in .SHP file to a raster geotiff.
    Computes the extents from the geotransform of the source image of the roi.
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

    # compute extents
    x_min = topleft_x
    y_min = topleft_y + y*rows
    x_max = topleft_x + x*cols
    y_max = topleft_y

    out_fn = os.path.splitext(os.path.basename(src))[0] + '.tif'

    # gdal command construction from variables
    rasterize_cmd = ['gdal_rasterize',
                     '-a', 'class',
                     '-a_nodata', '0',
                     '-ot', 'UInt16',
                     '-te',  # specify extent
                     str(x_min), str(y_min),
                     str(x_max), str(y_max),
                     '-ts',  # specify the number of columns and rows
                     str(cols), str(rows),
                     '-l', os.path.splitext(os.path.basename(src))[0],  # layer name
                     src, out_fn]

    call(rasterize_cmd)

    return


def rasterize_ROI(data_path, geotrans, cols, rows, projection, dataset_format = 'MEM'):

    ds = gdal.OpenEx(data_path, gdal.OF_VECTOR)

    if ds is None:
        print 'Could not open ' + data_path
        sys.exit(1)

    layer = ds.GetLayer(0)
    driver = gdal.GetDriverByName(dataset_format)
    target_ds = driver.Create()

    return


def segment_features(segment_pixels):
    """For each band, compute: min, max, mean, variance, skewness, kurtosis"""
    features = []

    n_pixels, n_bands = segment_pixels.shape
    #n_pixels = segment_pixels[0] * segment_pixels[1]
    #n_bands = segment_pixels[2]

    for b in range(n_bands):
        stat = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stat.minmax) + list(stat)[2:]
        if n_pixels == 1:
            band_stats[3] = 0.0
        features += band_stats

    return features


def create_objects(image, roi_image, image_parameters):
    """
    http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
    reference for pairing the training segment with the image pixels

    https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb
    reference for creating objects
    :param image:
    :param roi_image:
    :return:
    """

    band_stack = []

    n_bands = image_parameters[2]

    # loop through each band

    for b in range(n_bands):
        img_band = image.GetRasterBand(b+1)
        band_stack.append(img_band.ReadAsArray())

    array_stack = np.dstack(band_stack)

    #print array_stack
    #print img_array
    #print band_stack

    roi_band = roi_image.GetRasterBand(1)
    roi_array = roi_band.ReadAsArray()

    #print array_stack[roi_array>0]
    # TODO: Create features
    labels = np.unique(roi_array[roi_array > 0])
    feature_id = np.unique(roi_array)

    #plt.figure()
    #io.imshow(roi_array, cmap=plt.cm.Spectral)
    #io.show()

    # create_objects
    objects = []
    object_ids = []

    for segment_label in labels:
        segment_pixels = array_stack[roi_array==segment_label]

        #print segment_label, ',', segment_pixels.shape
        #print segment_pixels

        segment_model = segment_features(segment_pixels)
        # TODO: Fix this section. Individual segments need to have their own ids. Not class labels
        # Read Training data-set definition in
        # https://www.machinalis.com/blog/obia/
        objects.append(segment_model)
        object_ids.append(segment_label)
        #print segment_model

    print "Created %i objects" % len(objects)

    return

if __name__ == "__main__":
    start = tm.time()

    img_dir = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\1b. LAND COVER CLASSIFICATION\\test_clip.tif"
    roi_dir = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\1b. LAND COVER CLASSIFICATION\WORKING FILES\\training sites\\training_sites.shp"

    # convert .shp file of training segments to .tif with parameters of image to be classified
    img = gdal.Open(img_dir, GA_ReadOnly)

    if img is None:
        print 'Could not open ' + img_dir
        sys.exit(1)

    # collect image parameters
    cols = img.RasterXSize
    rows = img.RasterYSize
    num_bands = img.RasterCount
    img_gt = img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_driver = img.GetDriver()

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    rasterize_roi(roi_dir, img_gt, cols, rows)

    for root, dirs, files in os.walk(os.getcwd()):
        if '1b.' in root:
            pass
        else:
            for f in files:
                if f == 'training_sites.tif':
                    #print f
                    roi = gdal.Open(f, GA_ReadOnly)
                    if img is None:
                        print 'Could not open ' + img_dir
                        sys.exit(1)

                    create_objects(img, roi, img_params)

                # the roi image uses has the same parameters as the satellite image

    print 'Processing time: %f' % (tm.time() - start)