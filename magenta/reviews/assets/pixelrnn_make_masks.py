# Author: Kyle Kastner
# License: MIT
#
# Based on masking code from Ishaan Gulrajani
# License: MIT
# https://github.com/igul222/pixel_rnn/blob/master/pixel_rnn.py
#
# Generates images for the Magenta Pixel Recurrent Neural Networks review
# https://github.com/tensorflow/magenta/blob/master/magenta/reviews/pixelrnn.md
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

n_channels_in_dataset = nc = 3
input_dim = 3
output_dim = 3
filter_size = 5
mask_type = 'B'
mask = np.ones((output_dim, input_dim, filter_size, filter_size),
               dtype="float32")

center = filter_size // 2

for i in range(filter_size):
    for j in range(filter_size):
        if (j > center) or (j == center and i > center):
            mask[:, :, j, i] = 0.

for i in range(nc):
    for j in range(nc):
        if (mask_type == 'A' and i >= j) or (mask_type == 'B' and i > j):
            mask[j::nc, i::nc, center, center] = 0.

f, axarr = plt.subplots(input_dim, output_dim)
s = {0: "B",
     1: "G",
     2: "R"}
for i in range(input_dim):
    for j in range(output_dim):
        axarr[i, j].imshow(mask[j, i], cmap="gray", interpolation="nearest",
                           extent=[0, 5, 5, 0])
        axarr[i, j].grid(True)
        if i == (input_dim - 1):
            axarr[i, j].set_xlabel(s[j], fontsize=20)
        if j == 0:
            # Add spaces to make it nicer looking
            h = axarr[i, j].set_ylabel(s[i] + "      ", rotation=0, fontsize=20)
        gray_rect = mpatches.Rectangle((2, 2), 1.0, 1.0,
                                       edgecolor="gray", facecolor="gray",
                                       alpha=0.4)
        axarr[i, j].add_patch(gray_rect)

f.text(0.5, 0.04, 'Channel in current layer', ha='center', va='center',
       fontsize=20)
f.text(0.06, 0.5, 'Channel in next layer', ha='center', va='center',
       rotation='vertical', fontsize=20)
# Do it last so it ends up on top...
labels = ["active (value 1)", "masked (value 0)", "center pixel"]
white_patch = mpatches.Patch(edgecolor="black", facecolor="white",
                             label=labels[0])
black_patch = mpatches.Patch(edgecolor="black", facecolor="black",
                             label=labels[1])
gray_patch = mpatches.Patch(edgecolor="gray", facecolor="gray",
                            alpha=0.4)
plt.figlegend(handles=[white_patch, black_patch, gray_patch], labels=labels,
              loc="upper center",
              bbox_to_anchor=(.5125, .9575))
plt.suptitle("Example masks for type '%s'" % mask_type, fontsize=28)
plt.show()
