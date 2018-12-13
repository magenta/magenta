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
"""Imports Music VAE model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_model import BaseDecoder
from .base_model import BaseEncoder
from .base_model import MusicVAE

from .configs import Config
from .configs import update_config

from .lstm_models import BaseLstmDecoder
from .lstm_models import BidirectionalLstmEncoder
from .lstm_models import CategoricalLstmDecoder
from .lstm_models import HierarchicalLstmDecoder
from .lstm_models import HierarchicalLstmEncoder
from .lstm_models import MultiOutCategoricalLstmDecoder
from .lstm_models import SplitMultiOutLstmDecoder
from .trained_model import TrainedModel
