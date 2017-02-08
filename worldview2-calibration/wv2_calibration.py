"""
The following workflow was derived from the technical note from Digital Globe:
Radiometric Use of Worldview-2 Imagery
"""

import sys
import gdal
import os
import time as tm
import numpy as np
from gdalconst import *
from subprocess import call


def toa_radiance(band_list, img_params):

    # create output calibrated worldview2 data-set
    cols = img_params[0]
    rows = img_params[1]
    bands = img_params[2]
    gt = img_params[3]
    proj = img_params[4]
    driver = img_params[5]

    out_ras = driver.Create('wv2_rad.tif', cols, rows, bands, GDT_Float32)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    # values from .IMD file that accompanied the image
    abs_cal_factors = {'BAND_B': 0.01260825,
                       'BAND_G': 0.009713071,
                       'BAND_R': 0.01103623,
                       'BAND_N': 0.01224380
                       }

    effective_bandwidth = {'BAND_B': 0.05430000,
                           'BAND_G': 0.06300000,
                           'BAND_R': 0.05740000,
                           'BAND_N': 0.09890000
                           }

    # iterate through each band and apply calibration values
    for band in enumerate(band_list):
        print "applying calibration values to %s..." %(band[1])
        no_value = band_list[band[1]].GetNoDataValue()
        out_band = out_ras.GetRasterBand(band[0] + 1)

        #print abs_cal_factors[band[1]]
        #print effective_bandwidth[band[1]]

        x_bsize = 1000
        y_bsize = 1000

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

                band_array = band_list[band[1]].\
                    ReadAsArray(j, i, num_cols, num_rows).\
                    astype(np.float32)

                # create no-value mask
                masked_ds = np.where(band_array == no_value, 0, band_array)

                # apply calibration values to image array
                #spec_rad = (masked_ds * abs_cal_factors[band[1]]) /\
                 #          effective_bandwidth[band[1]]
                #print cal_band

                # compute band-integrated radiance
                band_int_rad = abs_cal_factors[band[1]] * masked_ds

                # compute band-averaged spectral radiance
                band_ave_sr = band_int_rad / effective_bandwidth[band[1]]

                out_band.WriteArray(band_ave_sr, j, i)

        out_band.SetNoDataValue(no_value)
        out_band.FlushCache()
        out_band.GetStatistics(0, 1)


def earth_sun_distance():
    import math as m

    year = 2015
    month = 12
    day = 24
    uni_time = 2.0 + (10.0/60.0) + (12.502055/3600.0)

    # compute julian day
    a = int(year/100)
    b = 2 - a + int(a/4)

    jul_day = int(365.25*(year + 4716)) + int(30.6001 * (month + 1)) \
              + day + (uni_time/24.0) + (b - 1524.5)

    d = jul_day - 2451545.0
    g = (357.529 + 0.98560028 * d)
    es_dist = 1.00014 - (0.01671 * m.cos(g)) - (0.00014 * m.cos(2*g))

    return es_dist


def solar_geom():
    band_av_sri = {
        'BAND_B': 1974.2416,
        'BAND_G': 1865.4104,
        'BAND_R': 1559.4555,
        'BAND_N': 1069.7302
    }

    pass


def cal_ref():
    pass


def main():
    start = tm.time()

    fn = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\cmasked_ds.tif"
    img = gdal.Open(fn, GA_ReadOnly)

    if img is None:
        print 'Could not open ' + fn
        sys.exit(1)

    # collect image parameters
    cols = img.RasterXSize
    rows = img.RasterYSize
    num_bands = img.RasterCount
    img_gt = img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_driver = img.GetDriver()
    #proj_ref = img.GetProjectionRef()

    #print proj_ref

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    # band name-tags so that calibration factors don't get mixed up
    tag = ['BAND_B',
           'BAND_G',
           'BAND_R',
           'BAND_N']

    band_list = {}
    for band_num in range(len(tag)):
        #print band_num
        band_list[tag[band_num]] = img.GetRasterBand(band_num+1)

    #for i in band_list:
        #print i, ':', band_list[i]
        #print 'min: ', band_list[i].GetMinimum()
        #print 'max:', band_list[i].GetMaximum()
        #print 'band no.', band_list[i].GetBand(), i

    #toa_radiance(band_list, img_params)

    print earth_sun_distance()

    print 'Processing time: %f' % (tm.time()-start)


if __name__ == "__main__":
    main()