# Transform a video in a set of pairs

import argparse
import cv2
import os
import skvideo.io
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description='')
parser.add_argument('--video_in', dest='video_in', default='', help='frames folder', required=True)
parser.add_argument('--path_out', dest='path_out', default='./', help='Destination folder')
parser.add_argument('--offset', dest='offset', type=int, default=0, help='skip first frame to offset the output')
parser.add_argument('--skip', dest='skip', type=int, default=10, help='pair is created for translation')
parser.add_argument('--size', dest='size', type=int, default=256, help='size')

args = parser.parse_args()

def main(_):
    size = args.size
    video_filename = os.path.join(args.video_in)
    print "start parsing", video_filename
    vc = skvideo.io.VideoCapture(args.video_in)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
        print "didn't succed to open the file"

    i = 0
    while rval:
        i = i + 1
        rval, frame = vc.read()
        if rval and i < args.offset:
            continue
        if rval and (i-args.offset)%args.skip == 0:
            print "load frames", i, " and", i+1
            img_left = Image.fromarray(frame)
            if not rval:
                break
            i = i + 1
            rval, frame = vc.read()
            img_right = Image.fromarray(frame)

            # resize the images
            small_side = min(img_left.size)
            center = img_left.size[0]/2
            margin_left = center - small_side/2
            margin_right = margin_left + small_side

            img_left = img_left.crop((margin_left, 0, margin_right, small_side))
            img_left = img_left.resize((size, size), Image.ANTIALIAS)
            img_right = img_right.crop((margin_left, 0, margin_right, small_side))
            img_right = img_right.resize((size, size), Image.ANTIALIAS)

            # create the pair
            pair = Image.new('RGB', (size*2, size), color=0)
            pair.paste(img_left, (0,0))
            pair.paste(img_right, (size,0))

            # save file
            file_out = os.path.join(args.path_out, "p{:06d}.png".format(i))
            pair.save(file_out, "png")


if __name__ == '__main__':
    main(0)
