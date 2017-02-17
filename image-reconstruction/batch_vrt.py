import os
import glob
from subprocess import call

def list_raster(indir):

    tif_list = glob.glob(indir+'\*.tif')
    with open(indir+'\\tif_list.txt', 'wb') as f:
        for fn in tif_list:
            path, name = os.path.split(fn)
            print fn
            f.writelines(fn+'\n')
    return

def build_vrt(indir, outdir):
    
    list_dir = glob.glob(indir+'\*.txt')
    print '\nfound %s in %s' % (list_dir[0], indir)
    print '\nbuilding vrt...'

    # be careful when building vrt for landsat8 because
    # bands 10 and 11 becoming 2 and 3 in the vrt
    # do not forget to change the name of the output vrt
    ndvi_anom = outdir+"\\sub.vrt"
    vrt_make = ["gdalbuildvrt", "-separate", "-input_file_list", list_dir[0], ndvi_anom]
    call(vrt_make)
    
    return

def main():
    refimg_dir = r"reference image"
    subimg_dir = r"subject image"
    out_dir = os.getcwd()

    list_raster(refimg_dir)
    build_vrt(refimg_dir, refimg_dir)
    list_raster(subimg_dir)
    build_vrt(subimg_dir, subimg_dir)

if __name__ == "__main__":
    main()
