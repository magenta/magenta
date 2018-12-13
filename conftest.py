import sys

collect_ignore = []
if sys.version_info.major != 2:
    collect_ignore.append('magenta/models/score2perf/datagen_beam_test.py')
