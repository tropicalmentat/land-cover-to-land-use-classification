# references
# http://chris35wills.github.io/courses/pydata_stack/

from subprocess import call
import sys
import gdal
from gdalconst import *
import numpy as np
import time as tm
import scipy
import scipy.ndimage
import scipy.stats
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression


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


def output_ds(out_array, img_params, d_type=GDT_Unknown, fn='result.tif'):
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

    out_ras = driver.Create(fn, cols, rows, bands, d_type)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    for b in range(bands):
        if bands == 1:
            out_band = out_ras.GetRasterBand(b + 1)
            out_band.WriteArray(out_array)
            out_band.SetNoDataValue(-99)
            out_band.FlushCache()
            out_band.GetStatistics(0, 1)
        else:
            out_band = out_ras.GetRasterBand(b+1)
            out_band.WriteArray(out_array[:,:,b])
            out_band.SetNoDataValue(-99)
            out_band.FlushCache()
            out_band.GetStatistics(0, 1)

    return


def downscale_image(hires_img, lores_param, fn='resampled.tif'):
    """
    Takes in a high resolution image and resamples it
    to a lower resolution.
    The parameters are taken from a low resolution
    reference image that are used to compute the extent
    and resolution of the resampled data-set.
    """

    # collect columns, rows, extent, resolution and geotrans, and proj of img to be masked
    cols = lores_param[0]
    rows = lores_param[1]
    geotrans = lores_param[3]

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

    downsample_cmd = [
                      'gdalwarp',
                      '-ot', 'Float32',
                      '-r', 'average',  # use average as sampling algorithm
                      '-te',  # specify extent
                      str(x_min), str(y_min),
                      str(x_max), str(y_max),
                      '-ts',  # specify the number of columns and rows
                      str(cols), str(rows),
                      hires_img,
                      fn
                      ]

    call(downsample_cmd)

    return


def frac_veg_cover(ndvi):
    """
    ndvi_o and ndvi_s, denote the average values for 100%
    and 0% imperviousness respectively and are estimated
    individually for each urban area.
    """

    ndvi_o = 0.110901
    ndvi_s = 0.557617
    fr = np.square(np.divide((ndvi - ndvi_o),(ndvi_s-ndvi_o)))

    return fr


def map_impervious(ind_var_img, dep_var_img, iv_param, dv_param, fn='impervious_surfaces.tif'):
    """
    Performs sub-pixel mixture analysis of landsat for impervious surfaces.
    Uses a resampled (using the average algorithm in gdalwarp) worldview2
    classified land-cover image.
    -----------------------------------------------------------------------
    ind_var_img : reference impervious surface image, downsampled from a
    high-resolution land-cover image.

    """

    iv = ind_var_img.GetRasterBand(1)
    iv_novalue = iv.GetNoDataValue()
    iv_array = iv.ReadAsArray(0, 0).astype(np.float32)
    iv_array[iv_array == iv_novalue] = np.nan

    dv = dep_var_img.GetRasterBand(1)
    dv_novalue = dv.GetNoDataValue()
    dv_array = dv.ReadAsArray(0, 0).astype(np.float32)
    dv_array[dv_array == dv_novalue] = np.nan

    # mask each array with no-value of the other
    iv_masked = np.where(np.isnan(dv_array), np.nan, iv_array)

    dv_masked = np.where(np.isnan(iv_array), np.nan, dv_array)

    # flatten each array
    iv_flat = iv_masked[~np.isnan(iv_masked)]
    dv_flat = dv_array[~np.isnan(dv_masked)]

    # random sample of pixels
    sample_pixels = random.sample(zip(iv_flat, dv_flat), 2500)  # the sample size suggested by article

    training_list_x = []
    training_list_y = []

    for i in range(len(sample_pixels)):
        training_list_x.append(sample_pixels[i][0])
        training_list_y.append(sample_pixels[i][1])

    training_sample_x = np.array(training_list_x)
    training_sample_y = np.array(training_list_y)

    # train linear regression model using samples
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(training_sample_x, training_sample_y)
    p = np.polyfit(training_sample_x, training_sample_y, 2)
    print p
    # print 'slope: %f' % slope
    # print 'intercept: %f' % intercept
    # print 'R value: %f' % r_value
    # print 'p value: %f' % p_value
    # print 'error: %f' % std_err

    # create test samples
    test_sample_x = np.random.rand(training_sample_x.shape[0])
    # print test_sample_x

    linear_model = slope * test_sample_x + intercept
    poly_model = p[0]*test_sample_x**2 + p[1]*test_sample_x + p[2]

    # plot samples and regression line
    fig, ax = plt.subplots()
    plt.title('wv2 average imp. vs. landsat ndvi')
    ax.scatter(training_sample_x, training_sample_y, c='g', marker='.', alpha=.4)
    plt.plot(test_sample_x, linear_model, 'k--', lw=0.75)
    plt.plot(test_sample_x, poly_model, 'k.', lw=0.75)
    ax.set_ylabel('Reference IS %')
    ax.set_ylim([-0.1, 1])
    ax.set_xlim([-0.1, 1])
    ax.set_xlabel('landsat8 NDVI')
    ax.text(0.6, 0.6, "${:.6f}$".format(slope) + '$x$' + ' $+$ ' + "${:.6f}$".format(intercept))
    ax.text(0.6, 0.55, "$r^2$" + " = " + "${:.6f}$".format(r_value))

    plot_title = 'plot.png'
    plt.savefig(plot_title, dpi=250)

    return


def map_impervious_1(ind_var_img, dep_var_img, iv_param, dv_param, fn='impervious_surfaces.tif'):
    """
    Performs sub-pixel mixture analysis of landsat for impervious surfaces.
    Uses a resampled (using the average algorithm in gdalwarp) worldview2
    classified land-cover image.
    -----------------------------------------------------------------------
    ind_var_img : reference impervious surface image, downsampled from a
    high-resolution land-cover image.

    """

    iv = ind_var_img.GetRasterBand(1)
    iv_novalue = iv.GetNoDataValue()
    iv_array = iv.ReadAsArray(0, 0).astype(np.float32)
    iv_array[iv_array == iv_novalue] = np.nan

    dv_bandlist = []
    for band in range(dv_param[2]):
        dv = dep_var_img.GetRasterBand(band + 1)
        dv_novalue = dv.GetNoDataValue()
        dv_array = dv.ReadAsArray(0, 0).astype(np.float32)
        dv_array[dv_array == dv_novalue] = np.nan
        dv_bandlist.append(dv_array)

    dv_bandstack = np.dstack(dv_bandlist)
    # print dv_bandstack

    # mask each array with no-value of the other
    iv_masked = np.where(np.isnan(dv_bandstack[:, :, 0]), np.nan, iv_array)

    dv_masked = None
    for band in range(dv_bandstack.shape[2]):
        dv_masked = np.where(np.isnan(iv_array), np.nan, dv_bandstack[:, :, band])

    # flatten each array
    iv_flat = iv_masked[~np.isnan(iv_masked)]
    dv_flat = dv_bandstack[~np.isnan(dv_masked)]
    # print dv_flat

    # random sample of pixels
    # TODO random sampling of ndarray
    # training_list = []
    # for b in range(dv_param[2]):
    #     sample_b = zip(iv_flat, dv_flat[:,b])
    #     training_list.append(sample_b)
    # training_stack = np.dstack(training_list)
    # print len(training_stack)
    sample_pixels = random.sample(zip(iv_flat, dv_flat), 2500)  # the sample size suggested by article

    #
    training_list_x = []
    training_list_y = []
    #
    for i in range(len(sample_pixels)):
        training_list_x.append(sample_pixels[i][0])
        training_list_y.append(sample_pixels[i][1])

    training_sample_x = np.array(training_list_x)
    training_sample_y = np.array(training_list_y)
    print training_sample_y.shape

    # train linear regression model using samples
    # from sklearn.feature_selection import RFE
    lr = LinearRegression()
    # ft_sel = RFE(lr, 9)
    # ft_sel = ft_sel.fit(training_sample_y, training_sample_x)
    # print ft_sel.score
    # print ft_sel.support_
    # print ft_sel.ranking_

    from sklearn.feature_selection import SelectFromModel
    mod_sel = SelectFromModel(lr)
    mod_sel = mod_sel.fit_transform(training_sample_y, training_sample_x)
    print mod_sel.shape

    # lr.fit(training_sample_x.reshape(50,50), training_sample_y.reshape(50,50,9))
    #
    # p = np.polyfit(training_sample_x, training_sample_y, 2)
    # print p
    # # print 'slope: %f' % slope
    # # print 'intercept: %f' % intercept
    # # print 'R value: %f' % r_value
    # # print 'p value: %f' % p_value
    # # print 'error: %f' % std_err
    #
    # # create test samples
    # test_sample_x = np.random.rand(50,50)
    # predicted_y = lr.predict(test_sample_x)
    # print predicted_y
    # print test_sample_x
    #
    # poly_model = p[0]*test_sample_x**2 + p[1]*test_sample_x + p[2]
    #
    # # # plot samples and regression line
    # # fig, ax = plt.subplots()
    # # plt.title('wv2 average imp. vs. landsat ndvi')
    # # ax.scatter(training_sample_x, training_sample_y, c='g', marker='.', alpha=.4)
    # # plt.plot(test_sample_x, linear_model, 'k--', lw=0.75)
    # # plt.plot(test_sample_x, poly_model, 'k.', lw=0.75)
    # # ax.set_ylabel('Reference IS %')
    # # ax.set_ylim([-0.1, 1])
    # # ax.set_xlim([-0.1, 1])
    # # ax.set_xlabel('landsat8 NDVI')
    # # ax.text(0.6, 0.6, "${:.6f}$".format(slope) + '$x$' + ' $+$ ' + "${:.6f}$".format(intercept))
    # # ax.text(0.6, 0.55, "$r^2$" + " = " + "${:.6f}$".format(r_value))
    #
    # plot_title = 'plot.png'
    # plt.savefig(plot_title, dpi=250)
    #


    return


def main():
    img_dir = 'impervious-vegetation_postprocessed.tif'
    ndvi_dir = 'landsat_ndvi_masked.tif'

    ndvi_img = open_image(ndvi_dir)
    ndvi_param = get_img_param(ndvi_img)
    print 'The landsat NDVI image has \n{} columns \n{} rows'\
        .format(ndvi_param[0], ndvi_param[1])

    img = open_image(img_dir)
    img_param = get_img_param(img)
    print '\nThe worldview2 image has \n{} columns\n{} rows'\
        .format(img_param[0], img_param[1])

    print '\n{} {}'.\
        format(img_param[0]/float(ndvi_param[0]),img_param[1]/float(ndvi_param[1]))

    # downscale_image(img_dir, landsat_param)

    resampled_ = open_image('resampled.tif')
    resampled_param = get_img_param(resampled_)
    # map_impervious(resampled_, landsat_img, resampled_param, landsat_param)
    # map_impervious(landsat_img, resampled_, landsat_param, resampled_param)

    landsat_dir = "reflectance.tif"
    landsat_img = open_image(landsat_dir)
    landat_param = get_img_param(landsat_img)
    map_impervious_1(resampled_, landsat_img, resampled_param, landat_param)

if __name__=="__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)