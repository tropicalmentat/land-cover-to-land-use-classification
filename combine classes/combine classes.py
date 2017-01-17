import sys
import gdal
import os
import time as tm
import numpy as np
from gdalconst import *


def combine_class(band):

    classes = band.GetCategoryNames()
    color_int = band.GetRasterColorInterpretation()

    attr_table = band.GetDefaultRAT()
    cols = attr_table.GetColumnCount()
    rows = attr_table.GetRowCount()
    cat_names = band.GetCategoryNames()
    #print cat_names
    color_interpretation = band.GetRasterColorInterpretation()
    print 'band color interpretation is %s' % color_interpretation
    band.SetRasterColorInterpretation(GCI_PaletteIndex)
    print 'band color interpretation is set to %s' % color_int

    """The output class image from ArcGIS has no color table"""
    #color_table = band.GetColorTable()
    #print color_table
    #print color_table.GetColorEntryCount()

    # initialize new color table for image
    color_table = gdal.ColorTable(GPI_RGB)
    print 'color table palette interpretation is %s' % color_table.GetPaletteInterpretation()
    #print 'color table entry count is %s' % color_table.GetCount()

    color_entry = color_table.SetColorEntry(color_table, 255)
    #print color_entry(255,255,255,0)

    # prepare information from the raster attribute table for setting up color table
    class_value = []
    class_name = []
    rgb = []    # to be used for setting up class colors

    # get column headings
    #for col in range(cols):
        #print attr_table.GetNameOfCol(col)

    #print "\n"

    for row in range(rows):
        class_value.append(attr_table.GetValueAsString(row, 2))
        class_name.append(attr_table.GetValueAsString(row, 3))
        rgb.append((attr_table.GetValueAsString(row, 4),
                   attr_table.GetValueAsString(row, 5),
                   attr_table.GetValueAsString(row, 6)))

    #print class_name

    band.SetRasterCategoryNames(class_name)

    #for i in range(len(rgb)):
        # set color palette
        #print rgb[i][0]
        #red = color_table.GetColorEntry(int(rgb[i][0]))
        #print red
        #green = color_table.GetPaletteInterpretation(rgb[i][1])
        #blue = color_table.GetPaletteInterpretation(rgb[i][2])

        #color_table.SetColorEntry(i, red, green, blue)

        #print class_value[i], class_name[i], rgb[i]

    #band.FlushCache()

    return

def main():
    fn = "G:\LUIGI\ICLEI\IMAGE PROCESSING\\1b. LAND COVER CLASSIFICATION\WORKING FILES\\segmentation_spec20_spat20\\segments_svm3.tif"

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

    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]

    #print img.GetDescription()
    #print img.GetMetadataDomainList()
    #print img.GetMetadata()

    band = img.GetRasterBand(1)

    print combine_class(band)


if __name__ == "__main__":
    main()