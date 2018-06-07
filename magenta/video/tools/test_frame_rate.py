import argparse

import skvideo.io
import skvideo.datasets

import numpy as np

parser = argparse.ArgumentParser(description='')

parser.add_argument('--video_in', dest='video_in',
                    help="""one video or a path and a wildcard,
                    wildcard need to be inside a quote,
                    please note that ~ can be expanded only outside quote
                    for instance ~/test.'*' works, but '~/test.*' won't""", required=True)

args = parser.parse_args()

#vid_in = skvideo.io.FFmpegReader(args.video_in)
data = skvideo.io.ffprobe(args.video_in)['video']
rate = data['@r_frame_rate']


print eval(rate)
