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


def mask_subject(subject_stack, mask):
    """
    Masks the subject image-array
    :param subject_stack:
    :return:
    """

    # mask out cloud pixels
    # edit this into block reading
    masked = subject_stack[mask == 1]

    sub_shape = subject_stack.shape
    print sub_shape
    print masked.shape

    #plt.figure()
    #io.imshow(masked, cmap=plt.cm.Spectral)
    #io.show()

    return


def mask_reference(reference_stack, mask):
    """
    Masks the reference image-array
    :param reference_stack:
    :param mask:
    :return:
    """

    #mask out cloud pixels
    masked = reference_stack[mask == 1]

    return


def predict_dn():
    # create new array with shape of subject and reference scenes
    pass


def main():

    # Open subject and reference scenes and their corresponding
    # cloud and cloud shadow masks
    sub_dir = r"subject image\\sub.vrt"
    ref_dir = r"reference image\\ref.vrt"
    submask_dir = r"sub_mask.tif"
    refmask_dir = r"ref_mask.tif"

    sub_img = gdal.Open(sub_dir, GA_ReadOnly)
    if sub_img is None:
        print 'Could not open ' + sub_dir
        sys.exit(1)

    ref_img = gdal.Open(sub_dir, GA_ReadOnly)
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

    # build image array stacks for subject and reference scenes
    sub_bands = []
    sub_nbands = sub_img.RasterCount
    for band in range(sub_nbands):
        b = sub_img.GetRasterBand(band+1)
        b_array = b.ReadAsArray(0, 0)
        sub_bands.append(b_array)
        #print b_array

    sub_stack = np.dstack(sub_bands)
    #print sub_stack.shape

    ref_bands = []
    ref_nbands = ref_img.RasterCount
    for band in range(ref_nbands):
        b = ref_img.GetRasterBand(band + 1)
        b_array = b.ReadAsArray(0, 0)
        ref_bands.append(b_array)
        # print b_array

    ref_stack = np.dstack(sub_bands)
    #print ref_stack.shape

    # open subject and refence masks
    submask_array = submask_img.GetRasterBand(1).ReadAsArray(0, 0)
    refmask_array = refmask_img.GetRasterBand(1).ReadAsArray(0, 0)

    #plt.figure()
    #io.imshow(refmask_array) #cmap=plt.cm.Spectral)
    #io.show()

    mask_subject(sub_stack, submask_array)
    mask_reference(ref_stack, refmask_array)

if __name__ == "__main__":
    main()
