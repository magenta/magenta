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

// 1-based
const PIANO_CHANNEL = 1
const DRUM_CHANNEL = 10

class Midi extends events.EventEmitter{
	constructor(magenta){
		super()

		this._isEnabled = false

		this._magenta = magenta

		this._metronomeEnabled = false

		WebMidi.enable((err) => {
			if (!err){
				this._isEnabled = true

				this._magenta.instances().forEach((instance) => {
					this._magenta.updatePort(WebMidi.getOutputByName(instance.portName()))
				})

				if (WebMidi.inputs){
					WebMidi.inputs.forEach((input) => this._bindInput(input))
				}
				WebMidi.addListener('connected', (device) => {
					if (device.input) {
						this._bindInput(device.input)
					}
					if (device.output) {
						this._magenta.updatePort(device.output)
					}
				})
			}
		})
	}

	_bindInput(inputDevice){
		if (this._isEnabled){
			if (inputDevice.name == this._magenta.clockPortName()) {
				this._fromClock = inputDevice
				console.info('Connected to clock on port ' + inputDevice.name)
				this._fromClock.addListener(
						'controlchange', 'all', (event) => {
							this.emit('metronomeTick', event.value == 0 ? 0 : 1)
						})
				return
			}

			var isMagentaIn = this._magenta.fromPortName() == inputDevice.name

			if (isMagentaIn) {
				console.info('Adding Magenta Input: ' + inputDevice.name)
			} else {
				console.info('Adding External Input: ' + inputDevice.name)
			}

			WebMidi.addListener('disconnected', (device) => {
				if (device.input){
					console.info('Removing Disconnected Input: ' + device.name)
					device.input.removeListener('noteOn')
					device.input.removeListener('noteOff')
				}
			})
			inputDevice.addListener('noteon', 'all', (event) => {
				try {
					this.emit('keyDown', event.note.number, undefined, isMagentaIn,
										event.channel == DRUM_CHANNEL)
				} catch(e){
					console.warn(e)
				}
			})
			inputDevice.addListener('noteoff',  'all', (event) => {
				try {
					this.emit('keyUp', event.note.number, undefined, isMagentaIn,
										event.channel == DRUM_CHANNEL)
				} catch(e){
					console.warn(e)
				}
			})
		}
	}
}

export {Midi}