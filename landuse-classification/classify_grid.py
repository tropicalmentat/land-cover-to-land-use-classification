# issues

# for saving .prj for output geodataframe
# https://gis.stackexchange.com/questions/204201/geopandas-to-file-saves-geodataframe-without-coordinate-system
# https://github.com/geopandas/geopandas/issues/363

# to save classification report
# http://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

import gdal
import sys
import os
import shutil
import warnings
import time as tm
import pandas as pd
import geopandas as gpd
import scipy.stats as st
import numpy as np
from gdalconst import *
from subprocess import call
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

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
    # TODO raise exception if left parameter is not a geodataframe

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

    grouped_trainxi = []  # indices of training samples
    grouped_testxi = []  # indices of test samples

    # splits the samples into two sets by
    # performing random sampling on half the samples
    # then
    # uses set operations to determine the indices
    # of the other half for model testing
    for label, group in train_grid.groupby(by='lu_type'):
        sxi = group.sample(frac=0.5).index
        grouped_trainxi.append(list(sxi))
        txi = group.index.difference(sxi)
        grouped_testxi.append(list(txi))

    unpacked_trainxi = []
    unpacked_testxi = []
    for group in grouped_trainxi:
        for xi in group:
            unpacked_trainxi.append(xi)

    for group in grouped_testxi:
        for xi in group:
            unpacked_testxi.append(xi)

    return pd.Index(unpacked_trainxi), pd.Index(unpacked_testxi)


def classify_land_use(objects, grid, exp_trials=1):
    """
    Classifies land use.
    objects: GeoDataFrame of the polygon objects
    grid: DataFrame containing data for classification
    """

    for trial in range(1, exp_trials+1):
        print '========='
        print 'trial {}'.format(trial)
        print '========='
        # create directory for experiment trial
        xp_path = os.getcwd() + '\\trial_' + str(trial)
        print 'creating directory for trial {}...'.format(trial)

        if os.path.exists(xp_path):
            print 'path exists! deleting path'
            shutil.rmtree(xp_path)

        os.mkdir(xp_path)

        # retrieve training cells from grid
        training_grid = pd.merge(grid[grid['lu_code'] != 0], objects.loc[:,'min':],
                           right_index=True, left_index=True)

        train_xi, test_xi = stratify_sample(training_grid)

        training_x = training_grid.ix[train_xi].pivot(index='lu_type', columns='Id').stack().loc[:,'min':]
        labels = training_grid.ix[train_xi]['lu_type']

        x = objects.loc[:, 'min':]

        # list of classifiers
        names = [
                'neural_net',
                'decision_tree',
                'random_forest',
                'linear_svm'
                 ]

        clfs = [
                MLPClassifier(),
                DecisionTreeClassifier(),
                RandomForestClassifier(n_estimators=1000, oob_score=True),
                SVC(kernel='linear')
               ]

        # create directory for accuracy reports
        ar_path = xp_path + '\\accuracy_reports'

        # if os.path.exists(ar_path):
        #     print 'path exists! deleting path'
        #     shutil.rmtree(ar_path)
        #
        os.mkdir(ar_path)

        for name, clf in zip(names, clfs):

            if name == 'neural_net':

                # iterate through different mlp classifier parameters
                mlp_act = [
                           'relu',
                           'logistic',
                           'tanh',
                           'identity'
                           ]

                solvers = [
                           'lbfgs',
                           'sgd',
                          ]

                for act_func in mlp_act:
                    for solver in solvers:
                        print '----------------------------------------------------------'
                        print '{} activation function and {} solver...'.format(act_func, solver)
                        classifier = MLPClassifier(activation=act_func, solver=solver)
                        classifier.fit(training_x, labels)
                        pred = pd.DataFrame(classifier.predict(x), index=objects.index, columns=['lu_type'])

                        # confusion matrix
                        pred_results = pred.ix[test_xi]['lu_type']
                        test_results = training_grid.ix[test_xi]['lu_type']

                        classes = labels.unique()

                        matrix = pd.DataFrame()

                        matrix['truth'] = test_results
                        matrix['predict'] = pred_results

                        conf_matrix = pd.crosstab(matrix['truth'], matrix['predict']
                                                  , margins=True)
                        # print conf_matrix
                        conf_matrix_fn = ar_path + '\\' + name + '_' + \
                                         act_func + '_' + solver + '.xlsx'

                        writer = pd.ExcelWriter(conf_matrix_fn)
                        conf_matrix.to_excel(writer, 'Sheet1')
                        writer.save()

                        cr = classification_report(test_results, pred_results, classes)
                        print cr
                        clf_grid = pd.merge(grid, pred, right_index=True, left_index=True)
                        clf_grid_path = xp_path + '\\classified_' + name + '_' + act_func + '_' + solver
                        clf_grid.to_file(clf_grid_path
                                    , driver='ESRI Shapefile')

                        print '----------------------------------------------------------'
            else:
                print '----------------------------------------------------------'
                print '{} classifier...'.format(name)
                classifier = clf
                classifier.fit(training_x, labels)
                pred = pd.DataFrame(classifier.predict(x), index=objects.index, columns=['lu_type'])

                # confusion matrix
                pred_results = training_grid.ix[test_xi]['lu_type']
                test_results = pred.ix[test_xi]['lu_type']
                classes = labels.unique()

                matrix = pd.DataFrame()

                matrix['truth'] = test_results
                matrix['predict'] = pred_results

                conf_matrix = pd.crosstab(matrix['truth'], matrix['predict']
                                          , margins=True)

                conf_matrix_fn = ar_path + '\\' + name + '.xlsx'

                writer = pd.ExcelWriter(conf_matrix_fn)
                conf_matrix.to_excel(writer, 'Sheet1')
                writer.save()

                # print conf_matrix
                cr = classification_report(test_results, pred_results, classes)
                print cr
                clf_grid = pd.merge(grid, pred, right_index=True, left_index=True)
                clf_grid_path = xp_path + '\\classified_' + name
                clf_grid.to_file(clf_grid_path,
                            driver='ESRI Shapefile')

                print '----------------------------------------------------------'
        print '**********************************************************'

    return


def main():
    img_dir = "resampled.tif"
    poly_grid_dir = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\2. LAND USE CLASSIFICATION\landuse grids\landuse_grid100_t2.shp"
    ras_grid_dir = "grid100.tif"

    img = open_image(img_dir)
    # img_param = get_img_param(img)

    lu_grid = open_image(ras_grid_dir)
    # lu_grid_param = get_img_param(lu_grid)

    # rasterize(poly_grid_dir, img_param)

    obj = create_objects(img, lu_grid)

    poly_grid = gpd.read_file(poly_grid_dir)

    study_area = cull_no_data(poly_grid, obj)

    classify_land_use(study_area, poly_grid, 3)


if __name__ == "__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)