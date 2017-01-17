import os
import glob
from subprocess import call


def clip_raster(fn, shp, outdir):

    path, name = os.path.split(fn)
    print '\nclipping ' + name + ' to ' + outdir
    clipped = outdir+'\clip_' + name.split('.')[0] + '.TIF'
    clip_cmd = ['gdalwarp',
                '-srcnodata',
                '-99',
                '-cutline',
                shp,
                '-crop_to_cutline',
                fn,
                clipped]
    call(clip_cmd)

    return

def main():
    work_dir = "G:\LUIGI\ICLEI\IMAGE PROCESSING\IMAGES\LC81140512015331LGN01"
    ras_list = glob.glob(work_dir+'\*.TIFF')
    clip_shp = "G:\LUIGI\ICLEI\IMAGE PROCESSING\IMAGES\\brgyb_1999_buffer.shp"

    for root, dirs, files in os.walk(work_dir):
        #print root
        for f in files:
            #print f
            if '.xml' in f:
                pass
            elif '.TIF' in f:
                print f
                file_path = root+'\\'+f
                clip_raster(file_path, clip_shp, root)

if __name__ == "__main__":
    main()
