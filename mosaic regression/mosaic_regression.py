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
from sklearn import tree
from matplotlib import pyplot as plt
from matplotlib import colors
from subprocess import call

io.use_plugin('matplotlib')
# to get matplotlib display images the following was taken from
# https://github.com/scikit-image/scikit-image/issues/599


def display_image(img_array):

    #img = exposure.rescale_intensity(img_array, in_range='image', out_range=(0, 255))
    img = img_array
    #print img.shape

    #i = img
    r = img[:, :, 4]
    g = img[:, :, 3]
    b = img[:, :, 2]

    rgb = np.dstack((r, g, b))

    plt.figure()
    io.imshow(r)
    io.show()

    return


def mask_dataset(img_ds, mask_ds, b_count):
    """
    Masks the subject image-array
    :param subject_stack:
    :return:
    """
    mask = mask_ds.GetRasterBand(1).ReadAsArray(0, 0)

    # mask out cloud pixels
    # TODO edit this into block reading
    masked_bands = []
    for band in range(b_count):
        b = img_ds.GetRasterBand(band+1)
        b_array = b.ReadAsArray(0, 0)
        masked_array = b_array * mask
        #print masked_array
        masked_bands.append(masked_array)

    masked_array = np.dstack(masked_bands)
    #print masked_array.shape
    clear_pixels = masked_array >= 0
    #print clear_pixels.shape

    matrix = masked_array[clear_pixels].\
        reshape(clear_pixels.shape[0], clear_pixels.shape[1],
                clear_pixels.shape[2])

    #print matrix.shape

    return matrix


def inverse_mask(img_ds, mask_ds, b_count):
    mask = mask_ds.GetRasterBand(1).ReadAsArray(0, 0)
    inverse = np.where(mask == 1, 0, 1)

    # mask out cloud pixels
    # TODO edit this into block reading
    masked_bands = []
    for band in range(b_count):
        b = img_ds.GetRasterBand(band + 1)
        b_array = b.ReadAsArray(0, 0)
        masked_array = b_array * inverse
        # print masked_array
        masked_bands.append(masked_array)

    masked_array = np.dstack(masked_bands)
    clear_pixels = masked_array >= 0
    # print clear_pixels.shape

    matrix = masked_array[clear_pixels]. \
        reshape(clear_pixels.shape[0], clear_pixels.shape[1],
                clear_pixels.shape[2])

    print matrix.shape

    return matrix


def pixels_to_predict(img_ds, mask_1, mask_2, b_count):
    sub_mask = mask_1.GetRasterBand(1).ReadAsArray(0, 0)
    ref_mask = mask_2.GetRasterBand(1).ReadAsArray(0, 0)

    inverse_mask_2 = np.where(ref_mask == 1, 0, 1)
    new_mask = sub_mask * inverse_mask_2

    masked_bands = []
    for band in range(b_count):
        b = img_ds.GetRasterBand(band + 1)
        b_array = b.ReadAsArray(0, 0)
        masked_array = b_array * new_mask
        # print masked_array
        masked_bands.append(masked_array)

    masked_array = np.dstack(masked_bands)

    clear_pixels = masked_array >= 0
    # print clear_pixels.shape

    matrix = masked_array[clear_pixels]. \
        reshape(clear_pixels.shape[0], clear_pixels.shape[1],
                clear_pixels.shape[2])

    return matrix


def build_regression(x, y):
    # create new array with shape of subject and reference scenes
    x_reshape = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    y_reshape = y.reshape(y.shape[0] * y.shape[1], y.shape[2])
    regress = tree.DecisionTreeRegressor()
    regress = regress.fit(x_reshape, y_reshape)

    return regress


def apply_regression(predict_pix, regressor):
    predict_pix_reshape = predict_pix.reshape(predict_pix.shape[0] * predict_pix.shape[1], predict_pix.shape[2])
    print predict_pix.shape
    y = regressor.predict(predict_pix_reshape)
    y_reshape =  y.reshape(predict_pix.shape)

    return y_reshape


def output_ds(out_array, img_params):
    # create output raster data-set
    cols = img_params[0]
    rows = img_params[1]
    bands = img_params[2]
    gt = img_params[3]
    proj = img_params[4]
    driver = img_params[5]

    out_ras = driver.Create('regression_results.tif', cols, rows, bands, GDT_UInt16)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    return


def main():

    # Open subject and reference scenes and their corresponding
    # cloud and cloud shadow masks
    sub_dir = r"subject image\\sub.vrt"
    ref_dir = r"reference image\\ref.vrt"
    submask_dir = r"sub_mask.tif"
    refmask_dir = r"ref_mask.tif"
    unionmask_dir = r"union_mask.tif"

    sub_img = gdal.Open(sub_dir, GA_ReadOnly)
    if sub_img is None:
        print 'Could not open ' + sub_dir
        sys.exit(1)

    ref_img = gdal.Open(ref_dir, GA_ReadOnly)
    if ref_img is None:
        print 'Could not open ' + ref_dir
        sys.exit(1)

    submask_img = gdal.Open(submask_dir, GA_ReadOnly)
    if submask_img is None:
        print 'Could not open ' + submask_dir
        sys.exit(1)

    refmask_img = gdal.Open(refmask_dir, GA_ReadOnly)
    if refmask_img is None:
        print 'Could not open ' + refmask_dir
        sys.exit(1)

    unionmask_img = gdal.Open(unionmask_dir, GA_ReadOnly)
    if unionmask_img is None:
        print 'Could not open ' + refmask_dir
        sys.exit(1)

    # image parameters for output data-set
    cols = sub_img.RasterXSize
    rows = sub_img.RasterYSize
    num_bands = sub_img.RasterCount
    img_gt = sub_img.GetGeoTransform()
    img_proj = sub_img.GetProjection()
    img_driver = sub_img.GetDriver()

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    # build image array stacks for subject and reference scenes

    sub_nbands = sub_img.RasterCount

    # mask subject scene
    subject_union = mask_dataset(sub_img, unionmask_img, sub_nbands)

    # mask reference scene with inverse of subject scene mask
    reference_union = mask_dataset(ref_img, unionmask_img, sub_nbands)

    #display_image(mask_subject)
    #display_image(mask_reference)

    # build regression tree
    model = build_regression(subject_union, reference_union)

    # apply regression tree

    # mask subject scene with its cloud/shadow mask and
    # the inverse of the ref scene cloud/shadow mask
    mask_subject = pixels_to_predict(sub_img, submask_img, refmask_img, sub_nbands)

    #display_image(mask_subject)

    result = apply_regression(mask_subject, model)

    display_image(result)



if __name__ == "__main__":
    main()
