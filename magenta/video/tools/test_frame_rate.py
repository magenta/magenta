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

import argparse

import skvideo.io
import skvideo.datasets

import numpy as np

parser = argparse.ArgumentParser(description='')

parser.add_argument(
    '--video_in',
    dest='video_in',
    help="""one video or a path and a wildcard,
                    wildcard need to be inside a quote,
                    please note that ~ can be expanded only outside quote
                    for instance ~/test.'*' works, but '~/test.*' won't""",
    required=True)

args = parser.parse_args()

#vid_in = skvideo.io.FFmpegReader(args.video_in)
data = skvideo.io.ffprobe(args.video_in)['video']
rate = data['@r_frame_rate']

print eval(rate)
