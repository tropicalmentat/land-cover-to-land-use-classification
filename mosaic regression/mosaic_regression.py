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
    pix = masked_array[masked_array > 0]
    print pix

    return pix


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
    pix = masked_array[masked_array > 0, :]

    return pix


def predict_dn(X, Y):
    # create new array with shape of subject and reference scenes
    regress = tree.DecisionTreeRegressor()
    regress = regress.fit(X, Y)

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

    # build image array stacks for subject and reference scenes

    sub_nbands = sub_img.RasterCount

    # mask subject scene
    mask_subject = mask_dataset(sub_img, unionmask_img, sub_nbands)
    # mask reference scene with inverse of subject scene mask
    mask_reference = mask_dataset(ref_img, unionmask_img, sub_nbands)
    #mask_union = mask_dataset()

    #display_image(mask_subject)
    #display_image(mask_reference)

    # apply regression
    predict_dn(mask_reference, mask_subject)


if __name__ == "__main__":
    main()
