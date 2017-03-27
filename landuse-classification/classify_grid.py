import gdal
import sys
import time as tm
import pandas as pd
import geopandas as gpd
import scipy.stats as st
import numpy as np
import os
import random
import warnings
from gdalconst import *
from subprocess import call
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

# warnings.filterwarnings("ignore")


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
    Creates statistical objects for classification by computing
    pixel statistics per grid cell.
    """

    img = image.GetRasterBand(1)
    img_novalue = img.GetNoDataValue()
    img_arr = img.ReadAsArray(0, 0)
    grid = grid.GetRasterBand(1)
    grid_arr = grid.ReadAsArray(0, 0)

    print 'Image array nodata value: {}'.format(img_novalue)

    img_shp = img_arr.shape
    grid_shp = grid_arr.shape
    # grid_shp = 9

    if img_shp == grid_shp:
        print "Image and grid have the same shape ({} columns, {} rows)".\
            format(img_shp[1], img_shp[0])
        print "Preparing grid and training data..."
    else:
        raise ValueError, 'Variable arrays have different shapes.'

    # mask out novalue pixels
    img_arr[img_arr>1] = np.nan

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

    print '{} cells will be classified.'.format(len(objects))

    return objects


def cull_no_data(gdf, df):
    """
    Extracts the study area by taking the intersection
    of geodataframe and the object dataframe.
    The geodataframe must be the left parameter for the merge function
    to return a merged geodataframe

    gdf - Geodataframe of the polygon grid
    df - dataframe of statistical objects for classification'
    Returns the merged geodataframe
    """

    study_area = gdf.merge(df, how='inner', left_on=['Id'], right_index=True)
    study_area.to_file('study_area', driver='ESRI Shapefile')

    return study_area


def stratify_sample(train_grid):
    """
    randomly stratify training sites for confusion matrix
    returns a tuple of Index objects, the left element
    for the training samples, the right element for testing
    to produce the confusion matrix
    """

    grouped_sxi = []
    grouped_txi = []

    for label, group in train_grid.groupby(by='lu_type'):
        sxi = group.sample(frac=0.5).index
        grouped_sxi.append(list(sxi))
        txi = group.index.difference(sxi)
        grouped_txi.append(list(txi))

    unpacked_sxi = []
    unpacked_txi = []
    for group in grouped_sxi:
        for xi in group:
            unpacked_sxi.append(xi)

    for group in grouped_txi:
        for xi in group:
            unpacked_txi.append(xi)

    return pd.Index(unpacked_sxi), pd.Index(unpacked_txi)


def classify_land_use(objects, grid):
    """
    Classifies land use.
    objects: GeoDataFrame of the polygon objects
    grid: DataFrame containing data for classification
    """

    # retrieve training cells from grid
    tr_grid = pd.merge(grid[grid['lu_code'] != 0], objects.loc[:,'min':],
                       right_index=True, left_index=True)

    sxi, txi = stratify_sample(tr_grid)

    tr_x = tr_grid.ix[sxi].pivot(index='lu_type', columns='Id').stack().loc[:,'min':]
    labels = tr_grid.ix[sxi]['lu_type']

    x = objects.loc[:, 'min':]

    # create training grid
    clf = MLPClassifier(activation='relu')
    clf = RandomForestClassifier()
    fit = clf.fit(tr_x, labels)
    pred = pd.DataFrame(clf.predict(x), index=objects.index, columns=['lu_type'])

    # confusion matrix
    pred_results = tr_grid.ix[txi]['lu_type']
    tr_results = pred.ix[txi]['lu_type']
    classes = labels.unique()

    cm = pd.DataFrame(confusion_matrix(pred_results, tr_results,
                           labels=classes),
                      index=classes, columns=classes)

    cr = classification_report(tr_results, pred_results, classes)
    ks = cohen_kappa_score(tr_results, pred_results, classes)
    print cm
    print cr
    print 'kappa score: {}'.format(ks)

    # print pred
    new = pd.merge(grid, pred, right_index=True, left_index=True)
    new.to_file('classified', driver='ESRI Shapefile')

    return #pred


def main():
    img_dir = "resampled.tif"
    poly_grid_dir = "landuse_grid100.shp"
    ras_grid_dir = "grid100.tif"

    img = open_image(img_dir)
    # img_param = get_img_param(img)

    lu_grid = open_image(ras_grid_dir)
    # lu_grid_param = get_img_param(lu_grid)

    # rasterize(poly_grid_dir, img_param)

    obj = create_objects(img, lu_grid)

    poly_grid = gpd.read_file(poly_grid_dir)

    study_area = cull_no_data(poly_grid, obj)

    lu = classify_land_use(study_area, poly_grid)


if __name__ == "__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)