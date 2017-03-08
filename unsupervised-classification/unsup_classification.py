# referenced from
# nbviewer.jupyter.org/gist/om-henners/
# http://geoinformaticstutorial.blogspot.com/2016/02/k-means-clustering-of-satellite-images.html

import gdal
from gdalconst import *
import scipy.cluster as cluster
import numpy as np
from sklearn.cluster import KMeans
import sys
import time as tm


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
    bands = 1  # ndvi image needs only one band
    gt = img_params[3]
    proj = img_params[4]
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    out_ras = driver.Create(fn, cols, rows, bands, d_type)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)

    out_band = out_ras.GetRasterBand(1)

    out_band.WriteArray(out_array)

    out_band.SetNoDataValue(-99)
    out_band.FlushCache()
    out_band.GetStatistics(0, 1)

    return


def build_band_stack(image_dataset, num_bands):
    band_list = []
    for band in range(num_bands):
        b = image_dataset.GetRasterBand(band+1)
        b_array = b.ReadAsArray(0, 0)
        band_list.append(b_array)

    band_stack = np.dstack(band_list)

    return band_stack


def main():
    img_dir = "landsat_urban.tif"
    img = open_image(img_dir)
    img_param = get_img_param(img)

    band_ds = build_band_stack(img, img_param[2])

    flat_ds = band_ds.reshape(band_ds.shape[0]*band_ds.shape[1],
                              band_ds.shape[2])

    input_ds = flat_ds.astype(np.double)

    # scipy kmeans
    centroid, label = cluster.vq.kmeans(input_ds, 5)
    code, distance = cluster.vq.vq(input_ds,centroid)
    ods = code.reshape(band_ds.shape[0],band_ds.shape[1])

    output_ds(ods, img_param, d_type=GDT_Int16, fn='kmeans_scipy.tif')

    # scikit kmeans
    k_means = KMeans(n_clusters=5)
    labels = k_means.fit_predict(flat_ds)
    ods1 = labels.reshape(band_ds.shape[0], band_ds.shape[1])

    output_ds(ods1, img_param, d_type=GDT_Int16, fn='kmeans_sklearn.tif')

    return


if __name__ == "__main__":
    start = tm.time()
    main()
    print '\nProcessing time: %f seconds' % (tm.time() - start)