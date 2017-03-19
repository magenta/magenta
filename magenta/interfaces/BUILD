# Copyright 2016 Google Inc. All Rights Reserved.
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

# This package group describes which set of packages should be able to depend
# on model generators.
#
# We use this visibility rule because most model code should be private to that
# model. We want to expose only the generators and we want them to be exposed
# only to our interfaces.
package_group(
    name = "generator_interfaces",
    packages = [
        "//magenta/interfaces/midi",
        # internal generator interfaces
    ]
)