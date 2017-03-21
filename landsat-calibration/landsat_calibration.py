# reference: https://landsat.usgs.gov/landsat-8-l8-data-users-handbook-section-5

import gdal
from gdalconst import *
import numpy as np
import time as tm
import sys
import pandas as pd
from math import cos

print '\nrescaling values for converting pixel DN to radiance: \n'
RADIO_RESCALE = pd.DataFrame({"RADMULT": [0.012586,
                                         0.012888,
                                         0.011877,
                                         0.010015,
                                        0.0061286,
                                        0.0015241,
                                       0.00051372,
                                         0.011334,
                                        0.0023952,],

                             "RADADD": [-62.93066,
                                        -64.44177,
                                        -59.38254,
                                        -50.07470,
                                        -30.64322,
                                        -7.62069,
                                        -2.56858,
                                        -56.67078,
                                        -11.97606,],
                              },
                             index=['1', '2', '3', '4', '5',
                                    '6', '7', '8', '9'])

print RADIO_RESCALE, '\n'

REFLECT_RESCALE = {"REFMULT": 0.000020000, "REFADD": -0.100000}

DOY = []


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
    bands = out_array.shape[2]
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


def dn_to_radiance(img, img_param):

    band_list = []
    for b in range(img_param[2]):
        band = img.GetRasterBand(b+1)
        nodata = band.GetNoDataValue()
        raw_dn = band.ReadAsArray(0, 0).astype(np.float32)
        raw_dn[raw_dn==nodata] = np.nan
        rad = RADIO_RESCALE['RADMULT'].loc[str(b+1)]*raw_dn + RADIO_RESCALE['RADADD'].loc[str(b+1)]
        band_list.append(rad)
        # print rad[~np.isnan(rad)].max()
        # print rad[~np.isnan(rad)].min()

    band_stack = np.dstack(band_list)
    output_ds(band_stack, img_param, d_type=GDT_Float32, fn='radiance.tif')

    return band_stack


def reflectance(img, img_param, sun_elev, norm=True):

    ref_list = []
    toa_ref_list = []
    for b in range(img_param[2]):
        band = img.GetRasterBand(b + 1)
        nodata = band.GetNoDataValue()
        raw_dn = band.ReadAsArray(0, 0).astype(np.float32)
        raw_dn[raw_dn == nodata] = np.nan
        ref = REFLECT_RESCALE['REFMULT'] * raw_dn + REFLECT_RESCALE['REFADD']
        ref_list.append(ref)
        # print ref[~np.isnan(ref)].max()
        toa_ref = ref/cos(sun_elev)
        toa_ref_list.append(toa_ref)
        # print toa_ref[~np.isnan(toa_ref)].max()
        # print toa_ref[~np.isnan(toa_ref)].min()

    ref_stack = np.dstack(ref_list)
    toa_ref_stack = np.dstack(toa_ref_list)
    output_ds(ref_stack, img_param, GDT_Float32, fn='reflectance.tif')
    output_ds(toa_ref_stack, img_param, GDT_Float32, fn='toa_reflectance.tif')

    if norm:
        normalize(ref_stack, img_param)

    return


def normalize(ref_bands, img_param):
    """
    This method of normalization was derived from
    Wu (2004) in Normalized spectral mixture analysis
    for monitoring urban composition using ETM+ imagery.
    ref_bands: reflectance pixels array of n-dimensions
    img_param: list of landsat image parameters
    """

    avg_ref = np.mean(ref_bands, 2, )
    # print avg_ref[~np.isnan(avg_ref)]
    norm_ref_list =[]
    for b in range(ref_bands.shape[2],5):
        norm_ref = np.multiply(np.divide(ref_bands[:,:,b], avg_ref), 100)
        # print norm_ref[~np.isnan(norm_ref)]
        # print norm_ref
        norm_ref_list.append(norm_ref)

    norm_stack = np.dstack(norm_ref_list)
    output_ds(norm_stack,img_param, d_type=GDT_Float32,
              fn='normalized_reflectance.tif')
    return


def main():
    img_dir = "landsat_urban_masked.tif"

    img = open_image(img_dir)
    img_param = get_img_param(img)

    radiance = dn_to_radiance(img, img_param)

    solar_el = 60.87663588

    refl = reflectance(img, img_param, solar_el)


if __name__=="__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)