# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""convert all files in a folder to jpg."""
from __future__ import print_function

import argparse
import glob
import ntpath
import os

from PIL import Image

PARSER = argparse.ArgumentParser(description='')
PARSER.add_argument(
    '--path_in',
    dest='path_in',
    default='',
    help='folder where the pictures are',
    required=True)
PARSER.add_argument(
    '--path_out', dest='path_out', default='./', help='Destination folder')
PARSER.add_argument(
    '--xsize', dest='xsize', type=int, default=0, help='horizontal size')
PARSER.add_argument(
    '--ysize',
    dest='ysize',
    type=int,
    default=0,
    help='vertical size, if crop is true, will use xsize instead')
PARSER.add_argument(
    '--delete',
    dest='delete',
    action='store_true',
    help='use this flag to delete the original file after conversion')
PARSER.set_defaults(delete=False)
PARSER.add_argument(
    '--crop',
    dest='crop',
    action='store_true',
    help='by default the video is cropped')
PARSER.add_argument(
    '--strech',
    dest='crop',
    action='store_false',
    help='the video can be streched to a square ratio')
PARSER.set_defaults(crop=True)

ARGS = PARSER.parse_args()


def convert2jpg(path_in, path_out, args):
  """Convert all file in a folder to jpg files.

  Args:
    path_in: the folder that contains the files to be converted
    path_out: the folder to export the converted files
    args: the args from the parser
      args.crop: a boolean, true for cropping
      args.delete: a boolean, true to remove original file
      args.xsize: width size of the new jpg
      args.ysize: height size of the new jpg

  Returns:
    nothing

  Raises:
    nothing
  """
  path = '{}/*'.format(path_in)
  print('looking for all files in', path)
  files = glob.glob(path)
  file_count = len(files)
  print('found ', file_count, 'files')

  i = 0
  for image_file in files:
    i += 1
    try:
      if ntpath.basename(image_file).split('.')[-1] in ['jpg', 'jpeg', 'JPG']:
        print(i, '/', file_count, '  not converting file', image_file)
        continue  # no need to convert
      print(i, '/', file_count, '  convert file', image_file)
      img = Image.open(image_file)
      # print('file open')
      if args.xsize > 0:
        if args.crop:
          args.ysize = args.xsize
          # resize the images
          small_side = min(img.size)
          center = img.size[0] / 2
          margin_left = center - small_side / 2
          margin_right = margin_left + small_side
          img = img.crop((margin_left, 0, margin_right, small_side))
        if args.ysize == 0:
          args.ysize = args.xsize
        img = img.resize((args.xsize, args.ysize), Image.ANTIALIAS)
      # save file
      # remove old path & old extension:
      basename = ntpath.basename(image_file).split('.')[0]
      filename = basename + '.jpg'
      file_out = os.path.join(path_out, filename)
      print(i, '/', file_count, '  save file', file_out)
      img.save(file_out, 'JPEG')
      if args.delete:
        print('deleting', image_file)
        os.remove(image_file)
    except:  # pylint: disable=bare-except
      print("""can't convert file""", image_file, 'to jpg :')

if __name__ == '__main__':
  convert2jpg(ARGS.path_in, ARGS.path_out, ARGS)
