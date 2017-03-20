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

class Midi extends events.EventEmitter{
	constructor(){
		super()

		this._isEnabled = false

		WebMidi.enable((err) => {
			if (!err){
				this._isEnabled = true
				if (WebMidi.inputs){
					WebMidi.inputs.forEach((input) => this._bindInput(input))
				}
				WebMidi.addListener('connected', (device) => {
					if (device.input){
						this._bindInput(device.input)
					}
				})
			}
		})
	}

	_bindInput(inputDevice){
		if (this._isEnabled){
			WebMidi.addListener('disconnected', (device) => {
				if (device.input){
					device.input.removeListener('noteOn')
					device.input.removeListener('noteOff')
				}
			})
			inputDevice.addListener('noteon', 'all', (event) => {
				try {
					this.emit('keyDown', event.note.number)
				} catch(e){
					console.warn(e)
				}
			})
			inputDevice.addListener('noteoff', 'all',  (event) => {
				try {
					this.emit('keyUp', event.note.number)
				} catch(e){
					console.warn(e)
				}
			})
		}
	}
}

export {Midi}