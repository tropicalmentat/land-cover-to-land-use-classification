import rasterio as rio
import time as tm
import numpy as np
import pysal as ps
import geopandas as gpd
import pandas as pd
import rasterstats as rs
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import logistic

warnings.filterwarnings("ignore")

def compute_sv(image, poly, aff):
    """
    Computes the spatial variance of pixels within each cell.

    reference for pixel neighbors
    https://github.com/pysal/pysal/blob/master/doc/source/users/tutorials/weights.rst

    """

    data = image.read(1)
    # dm = np.where(data < 3, data, np.nan)

    zs = rs.zonal_stats(poly, data, affine=aff)
    print zs

    return


def landuse_is_profile(image, grid, poly_grid):
    """
    http://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python

    Cumulative frequency plots of impervious surfaces
    per training grid cell.
    """


    data = image.read(1)
    grid = grid.read(1)
    pgrid = poly_grid
    tr_cell_i = pgrid[pgrid['lu_code']!= 0].index

    # isolate training grid cells
    for i in tr_cell_i:

        data[data>=1] = np.nan
        cell_pix = data[grid==i]
        cell_pix = cell_pix[~np.isnan(cell_pix)]
        lu_type = pgrid.loc[i, 'lu_type']

        if len(cell_pix) > 0:
            values, base = np.histogram(cell_pix, bins=40, range=(0., 1.), normed=True)
            cf = np.cumsum(values)

            #########################################

            # scaling
            scaler = MinMaxScaler()
            cf_norm = scaler.fit_transform(cf)

            #########################################

            # logistic regression
            # print len(cf_norm), len(base[:-1])
            # print cf_norm, base[:-1]
            # log = LogisticRegression()
            # log.fit(cell_pix, np.cumsum(cell_pix))
            # log.predict(values)
            # logit = sm.Logit(cell_pix, np.cumsum(cell_pix))
            # result = logit.fit()
            # # print result.summary()

            #########################################

            # plot
            fig, ax = plt.subplots()
            ax.set_ylabel('Cumulative frequency')
            ax.set_xlabel('Proportion impervious surface')
            ax.set_ylim([0., 1.2])
            ax.set_xlim([0., 1.])
            plt.plot(base[:-1], cf_norm)
            plt.title(lu_type + ' ' + str(i))
            plt.show()

    return

def main():
    img_dir = 'resampled.tif'
    grid_dir = 'grid100.tif'
    sa_grid_dir = 'study_area\\study_area.shp'

    img = rio.open(img_dir)
    # affine = img.affine
    # pgrid = gpd.read_file(pgrid_dir)
    # print pgrid
    grid = rio.open(grid_dir)
    pgrid_dir = gpd.read_file(sa_grid_dir)

    landuse_is_profile(img, grid, pgrid_dir)

    return


if __name__=="__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)