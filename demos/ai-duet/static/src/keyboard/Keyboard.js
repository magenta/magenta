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
import {Midi} from 'keyboard/Midi'
import Buffer from 'Tone/core/Buffer'

class Keyboard extends events.EventEmitter{
	constructor(container){
		super()

		this._container = container

		this._active = false

		/**
		 * The audio key keyboard
		 * @type {AudioKeys}
		 */
		this._keyboard = new AudioKeys({polyphony : 88, rows : 1, octaveControls : false})
		this._keyboard.down((e) => {
			this.keyDown(e.note)
			this._emitKeyDown(e.note)
		})
		this._keyboard.up((e) => {
			this.keyUp(e.note)
			this._emitKeyUp(e.note)
		})

		/**
		 * The piano interface
		 */
		this._keyboardInterface = new KeyboardElement(container, 48, 2)
		this._keyboardInterface.on('keyDown', (note) => {
			this.keyDown(note)
			this._emitKeyDown(note)
		})
		this._keyboardInterface.on('keyUp', (note) => {
			this.keyUp(note)
			this._emitKeyUp(note)
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
		this._midi = new Midi()
		this._midi.on('keyDown', (note) => {
			this.keyDown(note)
			this._emitKeyDown(note)
		})
		this._midi.on('keyUp', (note) => {
			this.keyUp(note)
			this._emitKeyUp(note)
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

	_emitKeyDown(note){
		if (this._active){
			this.emit('keyDown', note)
		}
	}

	_emitKeyUp(note){
		if (this._active){
			this.emit('keyUp', note)
		}
	}

	keyDown(note, time=Tone.now(), ai=false){
		if (!this._active){
			return
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

	keyUp(note, time=Tone.now(), ai=false){
		if (!this._active){
			return
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

	_resize(){
		const keyWidth = 24
		let octaves = Math.round((window.innerWidth / keyWidth) / 12)
		octaves = Math.max(octaves, 2)
		octaves = Math.min(octaves, 7)
		let baseNote = 48
		if (octaves > 5){
			baseNote -= (octaves - 5) * 12
		}
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