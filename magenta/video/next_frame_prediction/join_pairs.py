# Join pairs
# Finds frames that matches and create pairs
#
# goal is to create pairs with a frame to the next fraime
# it can match a real frame to recursive generated frame
# for instance (r0001.png) with a real frame (f0002.png)

import argparse
import glob
import ntpath
import os
from PIL import Image, ImageDraw
from random import shuffle

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path_left', dest='path_left', default='',
                    help='folder for left pictures', required=True)
parser.add_argument('--path_right', dest='path_right', default='',
                    help='folder for right pictures', required=True)
parser.add_argument('--path_out', dest='path_out', default='./',
                    help='Destination folder')
parser.add_argument('--prefix', dest='prefix', default='p',
                    help='prefix to be used when genererating the pairs (f)')
parser.add_argument('--size', dest='size', type=int, default=-1,
                    help='resize the output')
parser.add_argument('--limit', dest='limit', type=int, default=-1,
                    help='cap the number of generated pairs')
args = parser.parse_args()

def is_match(l_name, r_list):
    basename = ntpath.basename(l_name)
    frame_number = int(basename.split('.')[0][1:])
    matched_name = '{:07d}.jpg'.format(frame_number+1)
    matches = [x for x in r_list if matched_name in x]
    if len(matches)>0:
        return matches[0], frame_number
    else:
        return False, 0

def main(_):
    size = args.size
    path = '{}/*.jpg'.format(args.path_left)
    print 'looking for recursive img in', path
    l_list = glob.glob(path)
    print 'found ', len(l_list), 'for left list'
    path = '{}/*.jpg'.format(args.path_right)
    print 'looking for frames img in', path
    r_list = glob.glob(path)
    print "found ", len(r_list), 'for right list'
    if args.limit > 0:
        shuffle(l_list)
        l_list = l_list[:args.limit]
    for left in l_list:
        match, i = is_match(left, r_list)
        if match:
            print "load left", left, " and right", match
            img_left = Image.open(left)
            img_right = Image.open(match)

            # resize the images
            if size == -1:
                size = min(img_left.size)
            img_left = img_left.resize((size, size), Image.ANTIALIAS)
            img_right = img_right.resize((size, size), Image.ANTIALIAS)

            # create the pair
            pair = Image.new('RGB', (size*2, size), color=0)
            pair.paste(img_left, (0,0))
            pair.paste(img_right, (size,0))

            # save file
            file_out = os.path.join(args.path_out,
                                    "{}{:07d}.jpg".format(args.prefix, i))
            pair.save(file_out, "JPEG")


if __name__ == '__main__':
    main(0)
