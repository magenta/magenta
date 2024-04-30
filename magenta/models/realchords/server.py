# Copyright 2024 The Magenta Authors.
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

"""Serves frontend and model endpoints for genjam interface.

@Author Alex Scarlatos (scarlatos@google.com)
"""

import json
import os

from absl import app as absl_app
from absl import flags
import flask

from magenta.models.realchords import agent_interface

PORT = flags.DEFINE_integer("port", 8080, "Port to listen on.")

app = flask.Flask(__name__, static_url_path="", static_folder="frontend")

agent: agent_interface.Agent = None


@app.get("/")
def index() -> str:
  """Get index page."""
  return flask.send_file(os.path.join("frontend", "index.html"))


@app.get("/models")
def get_models() -> str:
  """Get available model names."""
  assert agent is not None
  return json.dumps(agent.get_models())


@app.post("/play")
def play() -> str:
  """Generate new chords given context."""
  assert agent is not None
  payload = flask.request.get_json()
  new_chords, new_chord_tokens, intro_chord_tokens = agent.generate_live(
      payload["model"],
      payload["notes"],
      payload["chordTokens"],
      payload["frame"],
      payload["lookahead"],
      payload["commitahead"],
      float(payload["temperature"]),
      payload["silenceTill"],
      payload["introSet"],
  )
  return json.dumps({
      "newChords": new_chords,
      "newChordTokens": new_chord_tokens,
      "introChordTokens": intro_chord_tokens,
      "frame": payload["frame"],
  })


def main(_: list[str]) -> None:
  global agent
  agent = agent_interface.Agent()

  # TODO(alexscarlatos): start the server


if __name__ == "__main__":
  absl_app.run(main)
