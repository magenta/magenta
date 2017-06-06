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

const MAGENTA_METRONOME_CHANNEL = 2

const PIANO_CHANNEL = 1
const DRUM_CHANNEL = 10

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

		this._states = {}
		this._states[DRUM_CHANNEL] = {temperature: 64, isLooping: false, bundleIndex: 0, outChannel: PIANO_CHANNEL}
		this._states[PIANO_CHANNEL] = {temperature: 64, isLooping: false, bundleIndex: 0, outChannel: PIANO_CHANNEL}
		this._outChannel = PIANO_CHANNEL

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
		})
	}

	setOutChannel(drum) {
		if (drum) {
			this._outChannel = DRUM_CHANNEL
		} else {
			this._outChannel = PIANO_CHANNEL
		}
	}

	sendKeyDown(note, drum) {
		this._toMagenta.playNote(note, this._outChannel)
	}

	sendKeyUp(note, drum) {
		this._toMagenta.stopNote(note, this._outChannel)
	}

	adjustBundleIndex(amt){
		this._states[this._outChannel].bundleIndex = Math.min(Math.max(this._states[this._outChannel].bundleIndex + amt, 0), 127)
		console.info('Bundle Index: ' + this._states[this._outChannel].bundleIndex)
		this._toMagenta.sendControlChange(
			  BUNDLE_CC, this._states[this._outChannel].bundleIndex, this._outChannel)
	}

	toggleLoop() {
		this._states[this._outChannel].isLooping = !this._states[this._outChannel].isLooping
		console.info('Looping: ' + this._states[this._outChannel].isLooping)
		this._toMagenta.sendControlChange(
				LOOP_CC, this._states[this._outChannel].isLooping * 127, this._outChannel)
	}

	triggerMutate() {
		console.info('Mutate!')
		this._toMagenta.sendControlChange(MUTATE_CC, 127, this._outChannel)
	}

	triggerPanic() {
		console.info('PANIC!')
		this._toMagenta.sendControlChange(PANIC_CC, 127, this._outChannel)
	}

	adjustTemperature(amt) {
		this._states[this._outChannel].temperature = Math.min(Math.max(this._states[this._outChannel].temperature  + amt, 0), 127)
		console.info('Temperature: ' + this._states[this._outChannel].temperature)
		this._toMagenta.sendControlChange(
			  TEMPERATURE_CC, this._states[this._outChannel].temperature, this._outChannel)
	}

	setCallBars(num_bars) {
		console.info('Call Bars: ' + num_bars)
		this._toMagenta.sendControlChange(
			  LISTEN_TICKS_CC, num_bars, this._outChannel)
	}

	setResponseBars(num_bars) {
		console.info('Response Bars: ' + num_bars)
		this._toMagenta.sendControlChange(
			  RESPONSE_TICKS_CC, num_bars, this._outChannel)
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
			inputDevice.addListener(
					'noteon',
				  isMagentaIn ? [PIANO_CHANNEL, DRUM_CHANNEL] : 'all',
				  (event) => {
				try {
					this.emit('keyDown', event.note.number, undefined, isMagentaIn,
										event.channel == DRUM_CHANNEL)
				} catch(e){
					console.warn(e)
				}
			})
			inputDevice.addListener(
					'noteoff',
				  isMagentaIn ? [PIANO_CHANNEL, DRUM_CHANNEL] : 'all',
				  (event) => {
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