/**
 * Copyright 2016 Google Inc.
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
import WebMidi from 'webmidi'

const FROM_MAGENTA_PORT = 'magenta_out'
const TO_MAGENTA_PORT = 'magenta_in'

const BUNDLE_CC = 0
const LOOP_CC = 1
const MUTATE_CC = 2
const TEMPERATURE_CC = 3
const RESPONSE_TICKS_CC = 4
const LISTEN_TICKS_CC = 5
const PANIC_CC = 6

class Midi extends events.EventEmitter{
	constructor(){
		super()

		this._isEnabled = false

		WebMidi.enable((err) => {
			if (!err){
				this._isEnabled = true

				this._toMagenta = WebMidi.getOutputByName(TO_MAGENTA_PORT);
				if (!this._toMagenta) {
					console.error('Could not find magenta input port: ' + TO_MAGENTA_PORT)
				} else {
					console.info('Connected to magenta input port: ' + TO_MAGENTA_PORT)
				}

				if (WebMidi.inputs){
					WebMidi.inputs.forEach((input) => this._bindInput(input))
				}
				WebMidi.addListener('connected', (device) => {
					if (device.input){
						this._bindInput(device.input)
					}
					if (device.output && device.output.name == TO_MAGENTA_PORT){
						this._toMagenta = device.output
					}

				})
			}

			this._temperature = 64
			this._isLooping = false
			this._bundleIndex = 0
		})
	}

	sendKeyDown(note) {
		this._toMagenta.playNote(note)
	}

	sendKeyUp(note) {
		this._toMagenta.stopNote(note)
	}

	adjustBundleIndex(amt){
		this._bundleIndex = Math.min(Math.max(this._bundleIndex + amt, 0), 127)
		console.info('Bundle Index: ' + this._bundleIndex)
		this._toMagenta.sendControlChange(BUNDLE_CC, this._bundleIndex)
	}

	toggleLoop() {
		this._isLooping = !this._isLooping
		console.info('Looping: ' + this._isLooping)
		this._toMagenta.sendControlChange(LOOP_CC, this._isLooping * 127)
	}

	triggerMutate() {
		console.info('Mutate!')
		this._toMagenta.sendControlChange(MUTATE_CC, 127)
	}

	triggerPanic() {
		console.info('PANIC!')
		this._toMagenta.sendControlChange(PANIC_CC, 127)
	}

	adjustTemperature(amt) {
		this._temperature = Math.min(Math.max(this._temperature + amt, 0), 127)
		console.info('Temperature: ' + this._temperature)
		this._toMagenta.sendControlChange(TEMPERATURE_CC, this._temperature)
	}

	setCallBars(num_bars) {
		console.info('Call Bars: ' + num_bars)
		this._toMagenta.sendControlChange(LISTEN_TICKS_CC, num_bars)
	}

	setResponseBars(num_bars) {
		console.info('Response Bars: ' + num_bars)
		this._toMagenta.sendControlChange(RESPONSE_TICKS_CC, num_bars)
	}

	_bindInput(inputDevice){
		if (this._isEnabled){
			console.info('Adding port: ' + inputDevice.name)
			var isMagentaIn = inputDevice.name == "magenta_out"

			WebMidi.addListener('disconnected', (device) => {
				if (device.input){
					device.input.removeListener('noteOn')
					device.input.removeListener('noteOff')
				}
			})
			inputDevice.addListener('noteon', isMagentaIn ? 1 : 'all', (event) => {
				try {
					this.emit('keyDown', event.note.number, undefined, isMagentaIn)
				} catch(e){
					console.warn(e)
				}
			})
			inputDevice.addListener('noteoff', isMagentaIn ? 1 : 'all', (event) => {
				try {
					this.emit('keyUp', event.note.number, undefined, isMagentaIn)
				} catch(e){
					console.warn(e)
				}
			})
		}
	}
}

export {Midi}