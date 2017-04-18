import os

"""
Ensure that a directory exists
"""
def ensuredir(dirpath):
	if not os.path.isdir(dirpath):
		os.makedirs(dirpath)