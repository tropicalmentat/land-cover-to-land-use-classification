# The following script was derived from the journal article
# Cloud-Free Satellite Image Mosaics with
# Regression Trees and Histogram Matching

import gdal
import arcpy
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

io.use_plugin('matplotlib')
# to get matplotlib display images the following was taken from
# https://github.com/scikit-image/scikit-image/issues/599


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


def display_image(img_array):

    # For Landsat8, rescale the values to 0-255
    img = exposure.rescale_intensity(img_array, in_range='dtype', out_range='uint8')
    eq_hist = exposure.equalize_hist(img, nbins=10)

    out_img = eq_hist

    r = out_img[:, :, 4]
    g = out_img[:, :, 3]
    b = out_img[:, :, 2]

    rgb = np.dstack((r, g, b))

    plt.figure()
    io.imshow(rgb)
    io.show()

    return


def mask_dataset(img_ds, mask_ds, b_count):
    """
    Masks the subject image-array
    """

    mask = mask_ds.GetRasterBand(1).ReadAsArray(0, 0)

    # mask out cloud pixels
    # TODO edit this into block reading
    masked_bands = []
    for band in range(b_count):
        b = img_ds.GetRasterBand(band+1)
        b_array = b.ReadAsArray(0, 0)
        masked_array = b_array * mask
        masked_bands.append(masked_array)

    masked_array = np.dstack(masked_bands)
    clear_pixels = masked_array >= 0

    matrix = masked_array[clear_pixels].\
        reshape(clear_pixels.shape[0], clear_pixels.shape[1],
                clear_pixels.shape[2])

    return matrix


def pixels_to_predict(img_ds, mask_1, mask_2, b_count):
    """
    Masks subject scene with its cloud/shadow mask and
    the inverse of the reference scene cloud/shadow mask.
    """

    sub_mask = mask_1.GetRasterBand(1).ReadAsArray(0, 0)
    ref_mask = mask_2.GetRasterBand(1).ReadAsArray(0, 0)

    inverse_mask_2 = np.where(ref_mask == 1, np.array(0), np.array(1))
    new_mask = sub_mask * inverse_mask_2

    masked_bands = []
    for band in range(b_count):
        b = img_ds.GetRasterBand(band + 1)
        b_array = b.ReadAsArray(0, 0)
        masked_array = b_array * new_mask
        masked_bands.append(masked_array)

    masked_array = np.dstack(masked_bands)

    clear_pixels = masked_array >= 0

    matrix = masked_array[clear_pixels].reshape(
        clear_pixels.shape[0],
        clear_pixels.shape[1],
        clear_pixels.shape[2]
    )

    return matrix


def build_regression(x, y):
    """
    First flattens arrays which are then fed
    into the decision tree regressor
    """

    # flatten input image-arrays
    x_reshape = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    y_reshape = y.reshape(y.shape[0] * y.shape[1], y.shape[2])
    regress = tree.DecisionTreeRegressor()
    regress = regress.fit(x_reshape, y_reshape)

    #r_score = regress.score(x_reshape, y_reshape)
    #print r_score

    return regress


def apply_regression(predict_pix, regressor):
    """
    Takes in the independent pixel values in a
    flattened array in the subject scene to
    predict the dependent pixel values
    """
    # flatten image array of subject scene
    predict_pix_reshape = predict_pix.reshape(predict_pix.shape[0] * predict_pix.shape[1], predict_pix.shape[2])

    # predict dependent pixel values in the reference scene
    y = regressor.predict(predict_pix_reshape)
    y_reshape = y.reshape(predict_pix.shape)
    r_score = regressor.score(y, predict_pix_reshape)
    print r_score

    return y_reshape


def output_ds(out_array, img_params, fn='result.tif'):
    """
    Helper function.
    Writes new data-set into disk
    and saves output arrays in the data-set.
    """

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


def main():

    # Open subject and reference scenes and their corresponding
    # cloud and cloud shadow masks
    sub_dir = r"subject image\\sub.vrt"
    ref_dir = r"reference image\\ref.vrt"
    submask_dir = r"sub_mask.tif"
    refmask_dir = r"ref_mask.tif"
    unionmask_dir = r"union_mask.tif"

    sub_img = open_image(sub_dir)
    ref_img = open_image(ref_dir)
    submask_img = open_image(submask_dir)
    refmask_img = open_image(refmask_dir)
    unionmask_img = open_image(unionmask_dir)

    # image parameters for output data-set
    img_params = get_img_param(sub_img)

    sub_nbands = sub_img.RasterCount

    # mask subject scene
    subject_union = mask_dataset(sub_img, unionmask_img, sub_nbands)
    #display_image(subject_union)

    # mask reference scene with inverse of subject scene mask
    reference_union = mask_dataset(ref_img, unionmask_img, sub_nbands)

    # TODO: Implement overall error computation for each scene that underwent cloud removal

    # build regression tree model
    model = build_regression(subject_union, reference_union)

    # prepare independent pixel values from subject scene
    mask_subject = pixels_to_predict(sub_img, submask_img, refmask_img, sub_nbands)
    #display_image(mask_subject)

    # mask reference scene to prepare for mosaic
    mask_ref = mask_dataset(ref_img, refmask_img, sub_nbands)
    output_ds(mask_ref, img_params, 'masked_ref.tif')

    # predict pixel values of reference scene
    result = apply_regression(mask_subject, model)
    #display_image(result)

    output_ds(result, img_params, 'regression_results.tif')

    # TODO: Implement raster mosaicing using arcpy
    # convert numpy array to raster
    #results_raster = arcpy.NumpyArrayToRaster(result, value_to_nodata=0)

if __name__ == "__main__":
    start = tm.time()
    main()
    print 'Processing time: %f seconds' % (tm.time() - start)