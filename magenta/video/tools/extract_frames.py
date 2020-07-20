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

# Lint as: python2, python3
"""Transform one or multiple video in a set of frames.

Files are prefixed by a f followed by the frame number.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import sys

from PIL import Image
import six
import skvideo.io

PARSER = argparse.ArgumentParser(description="""
Transform one or multiple video in a set of frames.

Files are prefixed by a f followed by the frame number""")

PARSER.add_argument(
    '--video_in',
    dest='video_in',
    help="""one video or a path and a wildcard,
            wildcard need to be inside a quote,
            please note that ~ can be expanded only outside quote
            for instance ~/test.'*' works, but '~/test.*' won't""",
    required=True)
PARSER.add_argument(
    '--from',
    dest='from_s',
    type=float,
    default=-1,
    help='starting time in second (-1)')
PARSER.add_argument(
    '--to',
    dest='to_s',
    type=float,
    default=-1,
    help='last time in second (-1)')
PARSER.add_argument(
    '--path_out', dest='path_out', default='./', help='Destination folder (./)')
PARSER.add_argument(
    '--offset',
    dest='offset',
    type=int,
    default=0,
    help="""skip first frame to offset the output (0)
            useful with '--skip' to extract only a subset""")
PARSER.add_argument(
    '--skip',
    dest='skip',
    type=int,
    default=1,
    help='"--skip n" will extract every n frames (1)')
PARSER.add_argument(
    '--size',
    dest='size',
    type=int,
    default=256,
    help='size (256), this argument is used, only if cropped')
PARSER.add_argument(
    '--start',
    dest='start',
    type=int,
    default=0,
    help='starting number for the filename (0)')
PARSER.add_argument(
    '--multiple',
    dest='multiple',
    type=int,
    default=10000,
    help=
    '''if used with a wildcard (*),
    "multiple" will be added for each video (10000)'''
)
PARSER.add_argument(
    '--format', dest='format_ext', default='jpg', help='(jpg) or png')
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


def crop(img, size):
  """resize the images.

  Args:
    img: a pillow image
    size: the size of the image (both x & y)

  Returns:
    nothing
  """
  small_side = min(img.size)
  center = img.size[0] / 2
  margin_left = center - small_side / 2
  margin_right = margin_left + small_side
  img = img.crop((margin_left, 0, margin_right, small_side))
  img = img.resize((size, size), Image.ANTIALIAS)
  return img


def main(_):
  """The main fonction use skvideo to extract frames as jpg.

  It can do it from a part or the totality of the video.

  Args:
    Nothing
  """
  print('argument to expand', ARGS.video_in)
  print('argument expanded', glob.glob(ARGS.video_in))
  video_count = 0
  for video_filename in glob.glob(ARGS.video_in):
    print('start parsing', video_filename)
    data = skvideo.io.ffprobe(video_filename)['video']
    rate_str = six.ensure_str(data['@r_frame_rate']).split('/')
    rate = float(rate_str[0]) / float(rate_str[1])
    print('detected frame rate:', rate)

    print('load frames:')
    video = skvideo.io.vreader(video_filename)
    frame_count = 0
    file_count = 0
    for frame in video:
      if (frame_count > ARGS.offset) and \
         ((frame_count-ARGS.offset)%ARGS.skip == 0) and \
         (frame_count/rate >= ARGS.from_s) and \
         (frame_count/rate <= ARGS.to_s or ARGS.to_s == -1):
        print(frame_count,)
        img = Image.fromarray(frame)
        if ARGS.crop:
          img = crop(img, ARGS.size)
        # save file
        file_number = file_count + video_count * ARGS.multiple + ARGS.start
        if ARGS.format_ext.lower() == 'jpg':
          file_out = os.path.join(ARGS.path_out,
                                  'f{:07d}.jpg'.format(file_number))
          img.save(file_out, 'JPEG')
        elif ARGS.format_ext.lower() == 'png':
          file_out = os.path.join(ARGS.path_out,
                                  'f{:07d}.png'.format(file_number))
          img.save(file_out, 'PNG')
        else:
          print('unrecognize format', ARGS.format_ext)
          sys.exit()
        file_count += 1
      frame_count += 1
    video_count += 1


if __name__ == '__main__':
  main(0)
