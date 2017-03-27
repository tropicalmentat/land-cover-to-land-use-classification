import rasterio
import time as tm
import numpy as np
import pysal as ps
import geopandas as gpd
import rasterstats as rs


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


def main():
    img_dir = 'resampled.tif'
    pgrid_dir = 'study_area\\study_area.shp'

    img = rasterio.open(img_dir)
    affine = img.affine
    # pgrid = gpd.read_file(pgrid_dir)
    # print pgrid

    print img.width, img.height, img.indexes
    compute_sv(img, pgrid_dir, affine)


    return


if __name__=="__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)