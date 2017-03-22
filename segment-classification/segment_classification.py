import gdal
import sys
import time as tm
import pandas as pd
import geopandas as gpd
import scipy.stats as st
import numpy as np
# import fiona
# import numpy.ma as ma
import os
import warnings
from gdalconst import *
from subprocess import call
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

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


def rasterize(src, img_param, fn='result.tif'):
    """
    Converts a region of interest vector data-set in .SHP file to a raster geotiff.
    Computes the extents from the geotransform of the source image of the roi.
    """

    # collect extent, resolution and geotrans, and proj of img to be masked
    topleft_x = img_param[3][0]
    topleft_y = img_param[3][3]
    x = img_param[3][1]
    y = img_param[3][5]

    # compute extents
    x_min = topleft_x
    y_min = topleft_y + y*img_param[1]
    x_max = topleft_x + x*img_param[0]
    y_max = topleft_y

    # gdal command construction from variables
    rasterize_cmd = ['gdal_rasterize',
                     '-a','Id',
                     # '-a_nodata', '0',
                     '-ot', 'UInt16',
                     '-te',  # specify extent
                     str(x_min), str(y_min),
                     str(x_max), str(y_max),
                     '-ts',  # specify the number of columns and rows
                     str(img_param[0]), str(img_param[1]),
                     '-l', os.path.splitext(os.path.basename(src))[0],  # layer name
                     src, fn]

    call(rasterize_cmd)

    return


def create_objects(image, grid):
    """
    http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
    reference for pairing the training segment with the image pixels

    https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb
    reference for creating objects

    The image array and grid array must have the same shape.
    """

    img = image.GetRasterBand(1)
    img_novalue = img.GetNoDataValue()
    img_arr = img.ReadAsArray(0, 0)
    grid = grid.GetRasterBand(1)
    grid_arr = grid.ReadAsArray(0, 0)

    print 'image array has shape {}'.format(img_arr.shape)
    print 'grid array has shape {}'.format(grid_arr.shape)
    print 'image array nodata value: {}'.format(img_novalue)

    img_arr[img_arr>1] = np.nan  # mask out novalue pixels
    # mask = img_arr > 1
    # masked_arr = ma.array(img_arr, mask=mask)
    # print masked_arr
    cell_id = []
    cell = []

    # compute statistics for each object
    for i in np.unique(grid_arr):
        obj = img_arr[grid_arr==i]
        no_nan = obj[~np.isnan(obj)]
        # filter out cells that have less than the minimum
        # required pixels
        if len(no_nan) >= 9:
            result = st.describe(no_nan, nan_policy='omit')
            cell_id.append(i)
            stats = list(result.minmax) + list(result)[2:]
            cell.append(stats)

    objects = pd.DataFrame(cell, index=cell_id,
                           columns=['min', 'max', 'mean',
                                    'variance', 'skewness',
                                    'kurtosis'])

    return objects


def extract_studyarea(df_1, df_2):
    """
    Extracts the study area by taking the intersection
    of dataframes.
    The geodataframe must be the left parameter.
    """
    sa = pd.merge(df_1, df_2, how='inner', left_on=['Id'], right_index=True)
    # print sa
    sa.to_file('study_area', driver='ESRI Shapefile')

    return sa


def classify_land_use(sa, tr):
    """
    Classifies land use
    """

    ts = pd.merge(tr[tr['lu_code'] != 0], sa, how='inner', on='Id')
    # print ts
    # print ts.groupby(by='lu_type_x').size()
    # print ts['lu_type_x'].astype('category')
    # print ts['lu_type_x'].describe()
    X = ts.pivot(index='lu_type_x', columns='Id').stack().iloc[:,5:]
    # print X
    labels = ts['lu_type_x']
    # X_PRED = sa.pivot(index='lu_type_x', columns='Id').stack().iloc[:,5:]
    XP = sa.iloc[:,4:]
    # print XP

    # print objects
    # create traininng objects
    clf = MLPClassifier()
    # clf = RandomForestClassifier()
    fit = clf.fit(X, labels)
    pred = pd.DataFrame(clf.predict(XP), index=sa['Id'], columns=['class'])
    print pred
    # print len(pred)
    # print len(sa)
    # print pred.iloc[:,0].value_counts()

    new = pd.merge(sa, pred, right_index=True, left_index=True)
    # print len(new)
    new.to_file('classified', driver='ESRI Shapefile')

    return pred


def main():
    img_dir = "resampled.tif"
    poly_grid_dir = "landuse_grid100.shp"
    ras_grid_dir = "grid100.tif"

    img = open_image(img_dir)
    img_param = get_img_param(img)

    lu_grid = open_image(ras_grid_dir)
    lu_grid_param = get_img_param(lu_grid)

    # rasterize(poly_grid_dir, img_param)

    obj = create_objects(img, lu_grid)
    # print obj.index.values
    # print obj

    poly_gird = gpd.read_file(poly_grid_dir)
    # print poly_gird['Id'].values
    # print poly_gird

    study_area = extract_studyarea(poly_gird, obj)
    # poly_gird[poly_gird['lu_code'] != 0].to_file('tr', driver='ESRI Shapefile')
    # print poly_gird[poly_gird['lu_code']!=0]
    lu = classify_land_use(study_area, poly_gird)

    # cld = pd.merge(study_area, lu, left_index=True, right_index=True)
    # print cld
    # cld.to_file('study_area', driver='ESRI Shapefile')


if __name__ == "__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)