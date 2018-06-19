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

# Move, Copy or deletes files in a given range

import argparse
import glob
import ntpath
import os
from PIL import Image, ImageDraw
from shutil import copyfile, move

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--mode', dest='mode', default='move', help='(move), copy, delete')
parser.add_argument(
    '--path_in',
    dest='path_in',
    required=True,
    help='folder where the pictures are')
parser.add_argument(
    '--path_out', dest='path_out', default='./', help='Destination folder')
parser.add_argument('--from', dest='from_', type=int, default=0, help='from')
parser.add_argument('--to', dest='to_', type=int, default=0, help='to')
parser.add_argument(
    '--offset',
    dest='offset',
    type=int,
    default=0,
    help='offset the files with a value (0)')
args = parser.parse_args()


def frame_number(name):
    basename = ntpath.basename(name)
    return int(basename.split('.')[0][1:])


def is_in(name, from_, to_):
    fn = frame_number(name)
    return (fn >= from_ and fn <= to_)


def offset_basename(name, offset):
    basename = ntpath.basename(name)
    ext = basename.split('.')[1]
    prefix = basename[0]
    fn = int(basename.split('.')[0][1:])
    return '{}{%:7d}.{}'.format(prefix, fn + offset, ext)


def move_from_to(path_in, path_out, from_, to_, mode, offset):
    path = '{}/*'.format(path_in)
    print 'looking for all files in', path
    files = glob.glob(path)
    file_count = len(files)
    print 'found ', file_count, 'files'

    i = 0
    for file in files:
        i = i + 1
        if is_in(file, from_, to_):
            #            try:
            basename = ''
            if offset == 0:
                basename = ntpath.basename(file)
            else:
                basename = offset_basename(file, offset)
            file_out = os.path.join(path_out, basename)

            if args.mode == 'move':
                print i, '/', file_count, '  moving', file, 'to', file_out
                move(file, file_out)
            elif args.mode == 'copy':
                print i, '/', file_count, '  copying', file, 'to', file_out
                copyfile(file, file_out)
            elif args.mode == 'delete':
                print i, '/', file_count, '  deleting', file
                os.remove(file)
            else:
                print 'mode', args.mode, 'not recognized'


#           except:
#              print '''can't''', mode, 'file', file
        else:
            print i, '/', file_count, '  not touching file', file

if __name__ == '__main__':
    move_from_to(args.path_in, args.path_out, args.from_, args.to_, args.mode,
                 args.offset)
