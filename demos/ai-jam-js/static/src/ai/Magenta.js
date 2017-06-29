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
import events from 'events'

const CLOCK_PORT_NAME = 'magenta_clock'
const FROM_MAGENTA_PORT_NAME = 'magenta_out'
const TO_MAGENTA_PIANO_PORT_NAME = 'magenta_piano_in'
const TO_MAGENTA_DRUMS_PORT_NAME = 'magenta_drums_in'

const DRUM_MODELS = ['Drums']
const PIANO_MODELS = ['Attention', 'Pianoroll', 'Performance']

// 0-based
const CLOCK_CC = 1
const MIN_LISTEN_TICKS_CC = 3
const MAX_LISTEN_TICKS_CC = 4
const RESPONSE_TICKS_CC = 5
const TEMPERATURE_CC = 6
const BUNDLE_CC = 8
const LOOP_CC = 10
const PANIC_CC = 11
const MUTATE_CC = 12

class MagentaInstance {
  constructor(portName, notifier, models){
    this._portName = portName
    this._notifier = notifier
    this._models = models
    this._active = false

    this._temperature = 64,
    this._isLooping = false
    this._bundleIndex = 0
    this._soloMode = false
    this._callBars = 0
    this._responseBars = 0
  }

  portName() {
    return this._portName
  }

  models() {
    return this._models
  }

  active() {
    return this._active
  }

  setPort(port) {
    console.info('Connected to Magenta Output Port: ' + port.name)
    this._toMagenta = port
    this._active = true
  }

  sendKeyDown(note) {
    this._toMagenta.playNote(note)
  }

  sendKeyUp(note) {
    this._toMagenta.stopNote(note)
  }

  bundleIndex() {
    return this._bundleIndex
  }

  setBundleIndex(index){
    this._bundleIndex = index
    this._notifier.notify('<b>Model:</b> ' + this._models[index])
    this._toMagenta.sendControlChange(BUNDLE_CC, index)
  }

  isLooping() {
    return this._isLooping
  }

  toggleLoop() {
    this._isLooping = !this._isLooping
    if (this._isLooping) {
      this._notifier.notify('<b>Looping</b> enabled')
    } else {
      this._notifier.notify('<b>Looping</b> disabled')
    }
    this._toMagenta.sendControlChange(LOOP_CC, this._isLooping * 127)
  }

  triggerMutate() {
    this._notifier.notify('<b>Mutating Sequence</b>')
    this._toMagenta.sendControlChange(MUTATE_CC, 127)
  }

  triggerPanic() {
    this._notifier.notify('<b>Clearing Sequence</b>')
    this._toMagenta.sendControlChange(PANIC_CC, 127)
  }

  temperature() {
    return this._temperature
  }

  setTemperature(temp) {
    this._temperature = Math.min(Math.max(temp, 0), 127)
    var float_temp = 0.1 + (this._temperature / 127.) * 1.9
    this._notifier.notify('<b>Temperature:</b> ' + float_temp.toFixed(2))
    this._toMagenta.sendControlChange(TEMPERATURE_CC, this._temperature)
  }

  callBars() {
    return this._callBars
  }

  setCallBars(numBars) {
    this._callBars = numBars
    this._notifier.notify('<b>Call Bars:</b> ' + numBars)
    this._toMagenta.sendControlChange(MIN_LISTEN_TICKS_CC, numBars)
    this._toMagenta.sendControlChange(MAX_LISTEN_TICKS_CC, numBars)
  }

  responseBars() {
    return this._responseBars
  }

  setResponseBars(numBars) {
    this._responseBars = numBars
    this._notifier.notify('<b>Response Bars:</b> ' + numBars)
    this._toMagenta.sendControlChange(RESPONSE_TICKS_CC, numBars)
  }
}

class Magenta extends events.EventEmitter {
  constructor(notifier){
    super()
    this._instances = [
        new MagentaInstance(TO_MAGENTA_PIANO_PORT_NAME, notifier, PIANO_MODELS),
        new MagentaInstance(TO_MAGENTA_DRUMS_PORT_NAME, notifier, DRUM_MODELS)
    ]
    this._selected = 0
  }

  updatePort(port) {
    var allActive = true
    for (var i = 0; i < this.instances().length; ++i) {
      if (this.instance(i).portName() == port.name) {
        this.instance(i).setPort(port)
      }
      allActive &= this.instance(i).active()
    }
    if (allActive) {
      this.emit('active')
    }
  }

  fromPortName() {
    return FROM_MAGENTA_PORT_NAME
  }

  clockPortName() {
    return CLOCK_PORT_NAME
  }

  selectedIndex() {
    return this._selected
  }

  selected() {
    return this._instances[this._selected]
  }

  toggleSelected() {
    this._selected = (this._selected + 1) % this._instances.length
  }

  instance(i) {
    return this._instances[i]
  }

  instances() {
    return this._instances
  }
}

export {Magenta}
