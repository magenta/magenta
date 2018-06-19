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

# Transform one or multiple video in a set of frames
# files are prefixed by a f followed by the frame number

import argparse
import glob
import inspect
import os
import skvideo.io
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description=""""
Transform one or multiple video in a set of frames
files are prefixed by a f followed by the frame number""")

parser.add_argument(
    '--video_in',
    dest='video_in',
    help="""one video or a path and a wildcard,
                    wildcard need to be inside a quote,
                    please note that ~ can be expanded only outside quote
                    for instance ~/test.'*' works, but '~/test.*' won't""",
    required=True)
parser.add_argument(
    '--from',
    dest='from_s',
    type=float,
    default=-1,
    help='starting time in second (-1)')
parser.add_argument(
    '--to',
    dest='to_s',
    type=float,
    default=-1,
    help='last time in second (-1)')
parser.add_argument(
    '--path_out',
    dest='path_out',
    default='./',
    help='Destination folder (./)')
parser.add_argument(
    '--offset',
    dest='offset',
    type=int,
    default=0,
    help="""skip first frame to offset the output (0)
                    useful with '--skip' to extract only a subset""")
parser.add_argument(
    '--skip',
    dest='skip',
    type=int,
    default=1,
    help='"--skip n" will extract every n frames (1)')
parser.add_argument(
    '--size', dest='size', type=int, default=256, help='size (256)')
parser.add_argument(
    '--start',
    dest='start',
    type=int,
    default=0,
    help='starting number for the filename (0)')
parser.add_argument(
    '--multiple',
    dest='multiple',
    type=int,
    default=10000,
    help=
    'if used with a wildcard (*), "multiple" will be added for each video (10000)'
)
parser.add_argument(
    '--format', dest='format_ext', default='jpg', help='(jpg) or png')
parser.add_argument(
    '--crop',
    dest='crop',
    action='store_true',
    help='by default the video is cropped')
parser.add_argument(
    '--strech',
    dest='crop',
    action='store_false',
    help='the video can be streched to a square ratio')
parser.set_defaults(crop=True)

args = parser.parse_args()


def main(_):
    size = args.size
    print 'argument to expand', args.video_in
    print 'argument expanded', glob.glob(args.video_in)
    video_count = 0
    for video_filename in glob.glob(args.video_in):
        print "start parsing", video_filename
        data = skvideo.io.ffprobe(video_filename)['video']
        rate = float(eval(data['@r_frame_rate']))
        print "detected frame rate:", rate

        print "load frames:"
        vc = skvideo.io.vreader(video_filename)
        frame_count = 0
        file_count = 0
        for frame in vc:
            if (frame_count > args.offset) and \
               ((frame_count-args.offset)%args.skip == 0) and \
               (frame_count/rate >= args.from_s) and \
               (frame_count/rate <= args.to_s or args.to_s == -1):
                print frame_count,
                img = Image.fromarray(frame)
                # resize the images
                small_side = min(img.size)
                center = img.size[0] / 2
                margin_left = center - small_side / 2
                margin_right = margin_left + small_side
                if args.crop:
                    img = img.crop((margin_left, 0, margin_right, small_side))
                    img = img.resize((size, size), Image.ANTIALIAS)

                # save file
                file_number = file_count + video_count * args.multiple + args.start
                if args.format_ext.lower() == 'jpg':
                    file_out = os.path.join(args.path_out,
                                            "f{:07d}.jpg".format(file_number))
                    img.save(file_out, 'JPEG')
                elif args.format_ext.lower() == 'png':
                    file_out = os.path.join(args.path_out,
                                            "f{:07d}.png".format(file_number))
                    img.save(file_out, 'PNG')
                else:
                    print 'unrecognize format', args.format_ext
                    quit()
                file_count = file_count + 1
            frame_count = frame_count + 1
        video_count = video_count + 1


if __name__ == '__main__':
    main(0)
