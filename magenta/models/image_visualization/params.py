import numpy as np

# Describe Octaves
octaves = [{
    'scale': 1.4,
    'iter_n': 190,
    'start_sigma': .44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.4,
    'iter_n': 150,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.4,
    'iter_n': 150,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.4,
    'iter_n': 10,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }]

# Set background color of image to visualize on
background_color = np.float32([44, 44, 44])

#classes = [1,2,3,4,5]
classes = [460]
