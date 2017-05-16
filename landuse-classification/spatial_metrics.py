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

    zs = rs.zonal_stats(poly, data, affine=aff)
    print zs

    return


def landuse_is_profile(image, grid, poly_grid):
    """
    http://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
    http://matplotlib.org/examples/statistics/histogram_demo_cumulative.html

    Cumulative frequency plots of impervious surfaces
    per training grid cell.
    """

    data = image.read(1)
    grid = grid.read(1)
    pgrid = poly_grid
    tr_cell_i = pgrid[pgrid['lu_code']!= 0].index

    # print len(tr_cell_i)
    # isolate training grid cells
    for i, cell_i in enumerate(tr_cell_i):

        data[data>=1] = np.nan
        cell_pix = data[grid==cell_i]
        cell_pix = cell_pix[~np.isnan(cell_pix)] # filter out nodata pixels
        lu_type = pgrid.loc[cell_i, 'lu_type']

        # plt.subplot()
        if len(cell_pix) > 0:

            # scale cell pixels from (0,1)
            scaler = MinMaxScaler()
            cell_pix_norm = scaler.fit_transform(cell_pix)

            plt.figure()
            plt.subplot()
            n, bins, patches = plt.hist(cell_pix_norm, bins=50, histtype='step', normed=1,
                                   cumulative=True, facecolor='g', label='Empirical')

            #########################################

            # cumulative frequency plotting parameters
            mu = cell_pix_norm.mean()
            sigma = cell_pix_norm.std()

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
            # reference for multiple plots in a grid
            # http://stackoverflow.com/questions/9603230/how-to-use-matplotlib-tight-layout-with-figure

            y = mlab.normpdf(bins, mu, sigma).cumsum()
            y /= y[-1]
            y[np.isnan(y)] = 0.
            y_norm = scaler.fit_transform(y)

            plt.plot(bins, y_norm, 'k--', linewidth=1.5, label='Theoretical')
            # ax.set_ylabel('Cumulative frequency')
            # ax.set_xlabel('Proportion impervious surface')
            # ax.set_ylim([0., 1.2])
            # ax.set_xlim([-0.2, 1.])
            plt.title(str(lu_type) + ' ' + str(cell_i))
    plt.tight_layout()
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