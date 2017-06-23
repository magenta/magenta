/**
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'style/status.css'

class Status {
  constructor(container){
    this._container = document.createElement('div')
    this._container.id = 'status'
    container.appendChild(this._container)

    const notification = document.createElement('div')
    notification.id = 'notification'
    this._container.appendChild(notification)

    const message = document.createElement('div')
    message.id = 'message'
    notification.appendChild(message)
    this._message = message

    const metronome = document.createElement('div')
    metronome.id = 'metronome'
    this._container.appendChild(metronome)

    this._metronomeBeats = []
    for (var i = 0; i < 4; ++i) {
      const beat = document.createElement('input')
      beat.setAttribute('type', 'radio')
      beat.setAttribute('name', 'metronome')
      beat.setAttribute('disabled', 'true')
      metronome.appendChild(beat)
      this._metronomeBeats.push(beat)
    }
  }

  notify(msg){
    this._message.innerHTML = msg

    var that = this
    function removeNotification() {
      if (that._message.innerHTML == msg) {
        that._message.innerHTML = ''
      }
    }
    setTimeout(removeNotification, 2000);
  }

  incrementMetronome() {
    for (var i = 0; i < 4; ++i) {
      if (this._metronomeBeats[i].checked) {
        this._metronomeBeats[i].checked = false
        this._metronomeBeats[(i + 1) % 4].checked = true
        break;
      }
    }
  }

  setMetronome(beat) {
    this._metronomeBeats.forEach((b) => {b.checked = false})
    this._metronomeBeats[beat].checked = true
  }
}

export {Status}