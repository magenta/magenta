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

import AudioKeys from 'audiokeys'
import Tone from 'Tone/core/Tone'
import events from 'events'
import {KeyboardElement} from 'keyboard/Element'
import buckets from 'buckets-js'
import Buffer from 'Tone/core/Buffer'

class Keyboard extends events.EventEmitter{
	constructor(container, midi, magenta, notifier){
		super()

		this._container = container
		this._midi = midi
		this._magenta = magenta
		this._notifier = notifier

		this._active = false
		this._drumMode = false
		this._soloMode = false

		/**
		 * The audio key keyboard
		 * @type {AudioKeys}
		 */
		this._keyboard = new AudioKeys({polyphony : 88, rows : 1, octaveControls : false})
		this._keyboard.down((e) => {
			// The drums model only plays lower "notes".
			var note = e.note - this._drumMode * 24
			this.keyDown(note, undefined, false, this._drumMode)
			this._emitKeyDown(note, undefined, false, this._drumMode)
		})
		this._keyboard.up((e) => {
			// The drums model only plays lower "notes".
			var note = e.note - this._drumMode * 24
			this.keyUp(note, undefined, false, this._drumMode)
			this._emitKeyUp(note, undefined, false, this._drumMode)
		})

		/**
		 * The piano interface
		 */
		this._keyboardInterface = new KeyboardElement(container, 48, 2)
		this._keyboardInterface.on('keyDown', (note) => {
			this.keyDown(note, undefined, false, this._drumMode)
			this._emitKeyDown(note, undefined, false, this._drumMode)
		})
		this._keyboardInterface.on('keyUp', (note) => {
			this.keyUp(note, undefined, false, this._drumMode)
			this._emitKeyUp(note, undefined, false, this._drumMode)
		})

		window.addEventListener('resize', this._resize.bind(this))
		//size initially
		this._resize()

		//make sure they don't get double clicked
		this._currentKeys = {}

		//a queue of all of the events
		this._eventQueue = new buckets.PriorityQueue((a, b) => b.time - a.time)
		this._boundLoop = this._loop.bind(this)
		this._loop()

		const bottom = document.createElement('div')
		bottom.id = 'bottom'
		container.appendChild(bottom)

		//the midi input
		this._midi.on('keyDown', (note, time, ai, drum) => {
			if (this._drumMode == drum) {
				this.keyDown(note, time, ai, drum)
			}
			this._emitKeyDown(note, time, ai, drum)
		})
		this._midi.on('keyUp', (note, time, ai, drum) => {
			if (this._drumMode == drum) {
				this.keyUp(note, time, ai, drum)
			}
			this._emitKeyUp(note, time, ai, drum)
		})
	}

	_loop(){
		requestAnimationFrame(this._boundLoop)
		const now = Tone.now()
		while(!this._eventQueue.isEmpty() && this._eventQueue.peek().time <= now){
			const event = this._eventQueue.dequeue()
			event.callback()
		}
	}

	_emitKeyDown(note, time=Tone.now(), ai=false, drum=false){
		if (this._active){
			this.emit('keyDown', note, time, ai, drum)
		}
	}

	_emitKeyUp(note, time=Tone.now(), ai=false, drum=false){
		if (this._active){
			this.emit('keyUp', note, time, ai, drum)
		}
	}

	keyDown(note, time=Tone.now(), ai=false, drum=false){
		if (!this._active){
			return
		}
		if (!ai && !this._soloMode) {
			this._magenta.selected().sendKeyDown(note)
		}
		if (!this._currentKeys[note]){
			this._currentKeys[note] = 0
		}
		this._currentKeys[note] += 1
		this._eventQueue.add({
			time : time,
			callback : this._keyboardInterface.keyDown.bind(this._keyboardInterface, note, ai)
		})
	}

	keyUp(note, time=Tone.now(), ai=false, drum=false){
		if (!this._active){
			return
		}
		if (!ai && !this._soloMode) {
			this._magenta.selected().sendKeyUp(note)
		}
		//add a little time to it in edge cases where the keydown and keyup are at the same time
		time += 0.01
		if (this._currentKeys[note]){
			this._currentKeys[note] -= 1
			this._eventQueue.add({
				time : time,
				callback : this._keyboardInterface.keyUp.bind(this._keyboardInterface, note, ai)
			})
		}
	}

	toggleDrumMode() {
		this._drumMode = !this._drumMode
		this._keyboardInterface.panic(true)
		this._keyboardInterface.panic(false)
		if (this._drumMode) {
			this._notifier.notify('Switched to <b>Drums</b>')
		} else {
			this._notifier.notify('Switched to <b>Piano</b>')
		}
	}

	toggleSoloMode() {
		this._soloMode = !this._soloMode
		if (this._soloMode) {
			this._notifier.notify('<b>Solo Mode</b> enabled')
		} else {
			this._notifier.notify('<b>Solo Mode</b> disabled')
		}
	}

	_resize(){
		const keyWidth = 24
		let octaves = Math.round((window.innerWidth / keyWidth) / 12)
		octaves = Math.max(octaves, 2)
		octaves = Math.min(octaves, 7)
		let baseNote = 36
		this._keyboardInterface.resize(baseNote, octaves)
	}

	activate(){
		container.classList.add('focus')
		this._active = true
	}

	deactivate(){
		container.classList.remove('focus')
		this._active = false
	}
}

export {Keyboard}