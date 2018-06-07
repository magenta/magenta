# convert file to jpg

import argparse
import glob
import ntpath
import os
import sys
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path_in', dest='path_in', default='',
                    help='folder where the pictures are', required=True)
parser.add_argument('--path_out', dest='path_out', default='./',
                    help='Destination folder')
parser.add_argument('--xsize', dest='xsize', type=int, default=0,
                    help='horizontal size')
parser.add_argument('--ysize', dest='ysize', type=int, default=0,
                    help='vertical size, if crop is true, will use xsize instead')
parser.add_argument('--delete', dest='delete', action='store_true',
                    help='use this flag to delete the orginal file after conversion')
parser.set_defaults(delete=False)
parser.add_argument('--crop', dest='crop', action='store_true',
                    help='by default the video is cropped')
parser.add_argument('--strech', dest='crop', action='store_false',
                    help='the video can be streched to a square ratio')
parser.set_defaults(crop=True)

args = parser.parse_args()

def convert2jpg(path_in, path_out, size):
    path = '{}/*'.format(path_in)
    print 'looking for all files in', path
    files = glob.glob(path)
    file_count = len(files)
    print 'found ', file_count, 'files'

    i = 0
    for file in files:
        i = i + 1
        try:
            if ntpath.basename(file).split('.')[-1] in [ 'jpg', 'jpeg', 'JPG' ]:
                print i, '/', file_count, '  not converting file', file
                continue  # no need to convert
            print i, '/', file_count, '  convert file', file
            img = Image.open(file)
            #print 'file open'
            if args.xsize > 0:
                if args.crop:
                    args.ysize = args.xsize
                    # resize the images
                    small_side = min(img.size)
                    center = img.size[0]/2
                    margin_left = center - small_side/2
                    margin_right = margin_left + small_side
                    img = img.crop((margin_left, 0, margin_right, small_side))
                if args.ysize == 0:
                    args.ysize = args.xsize
                img = img.resize((args.xsize, args.ysize), Image.ANTIALIAS)
            # save file
            basename = ntpath.basename(file).split('.')[0]  # remove old path & old extension
            filename = basename + '.jpg'
            file_out = os.path.join(path_out, filename)
            print i, '/', file_count, '  save file', file_out
            img.save(file_out, 'JPEG')
            if args.delete:
                print 'deleting', file
                os.remove(file)
        except Exception as e:
            print '''can't convert file''', file, 'to jpg :', str(e)


if __name__ == '__main__':
    convert2jpg(
        args.path_in,
        args.path_out,
        args.xsize)  #todo(dh) add ysize etc
