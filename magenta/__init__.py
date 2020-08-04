# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Pulls in all magenta libraries that are in the public API.."""

import magenta.common.beam_search
import magenta.common.concurrency
import magenta.common.nade
import magenta.common.sequence_example_lib
import magenta.common.state_util
import magenta.common.testing_lib
import magenta.common.tf_utils
import magenta.pipelines.dag_pipeline
import magenta.pipelines.drum_pipelines
import magenta.pipelines.lead_sheet_pipelines
import magenta.pipelines.melody_pipelines
import magenta.pipelines.note_sequence_pipelines
import magenta.pipelines.pipeline
import magenta.pipelines.pipelines_common
import magenta.pipelines.statistics
import magenta.version
from magenta.version import __version__
