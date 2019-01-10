# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Join pairs Finds frames that matches and create pairs.

The goal is to create pairs with a frame to the next frame
it can match a real frame to a recursively generated frame
for instance (r0001.png) with a real frame (f0002.png)
"""
from __future__ import print_function

import argparse
import glob
import ntpath
import os
from random import shuffle

from PIL import Image

PARSER = argparse.ArgumentParser(description='')
PARSER.add_argument(
    '--path_left',
    dest='path_left',
    default='',
    help='folder for left pictures',
    required=True)
PARSER.add_argument(
    '--path_right',
    dest='path_right',
    default='',
    help='folder for right pictures',
    required=True)
PARSER.add_argument(
    '--path_out', dest='path_out', default='./', help='Destination folder')
PARSER.add_argument(
    '--prefix',
    dest='prefix',
    default='p',
    help='prefix to be used when genererating the pairs (f)')
PARSER.add_argument(
    '--size', dest='size', type=int, default=-1, help='resize the output')
PARSER.add_argument(
    '--limit',
    dest='limit',
    type=int,
    default=-1,
    help='cap the number of generated pairs')
ARGS = PARSER.parse_args()


def is_match(l_name, r_list):
  """for a given frame, find the next one in a list of frame.

  Args:
    l_name: the name of file
    r_list: a list of potential file to be matched

  Returns:
    a match (or False if no match)
    the frame number of the match
  """
  basename = ntpath.basename(l_name)
  frame_number = int(basename.split('.')[0][1:])
  matched_name = '{:07d}.jpg'.format(frame_number + 1)
  matches = [x for x in r_list if matched_name in x]
  if matches:
    return matches[0], frame_number
  return False, 0


def main(_):
  """match pairs from two folders.

  it find frames in a folder, try to find a matching frame in an other folder,
  and build a pair.

  """
  size = ARGS.size
  path = '{}/*.jpg'.format(ARGS.path_left)
  print('looking for recursive img in', path)
  l_list = glob.glob(path)
  print('found ', len(l_list), 'for left list')
  path = '{}/*.jpg'.format(ARGS.path_right)
  print('looking for frames img in', path)
  r_list = glob.glob(path)
  print('found ', len(r_list), 'for right list')
  if ARGS.limit > 0:
    shuffle(l_list)
    l_list = l_list[:ARGS.limit]
  for left in l_list:
    match, i = is_match(left, r_list)
    if match:
      print('load left', left, ' and right', match)
      img_left = Image.open(left)
      img_right = Image.open(match)

      # resize the images
      if size == -1:
        size = min(img_left.size)
      img_left = img_left.resize((size, size), Image.ANTIALIAS)
      img_right = img_right.resize((size, size), Image.ANTIALIAS)

      # create the pair
      pair = Image.new('RGB', (size * 2, size), color=0)
      pair.paste(img_left, (0, 0))
      pair.paste(img_right, (size, 0))

      # save file
      file_out = os.path.join(ARGS.path_out, '{}{:07d}.jpg'.format(
          ARGS.prefix, i))
      pair.save(file_out, 'JPEG')


if __name__ == '__main__':
  main(0)
