"""Staff detection.

Holds the staff detector classes that can be used as part of an OMR pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.models.omr.staves import hough
from magenta.models.omr.staves import projection

# Alias the staff detectors to access them directly from the staves module.
# pylint: disable=invalid-name
FilteredHoughStaffDetector = hough.FilteredHoughStaffDetector
ProjectionStaffDetector = projection.ProjectionStaffDetector

# The default staff detector that should be used in production.
StaffDetector = FilteredHoughStaffDetector
