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
import Tone from 'Tone/core/Tone'
import 'style/tutorial.css'

const beat = 0.4
const testMelody = [
	{
		note : 60,
		time : beat * 0, 
		duration : beat
	},
	{
		note : 62,
		time : beat * 1,
		duration : beat
	},
	{
		note : 64,
		time : beat * 2,
		duration : beat
	},
	{
		note : 64,
		time : beat * 3,
		duration : beat * 0.5
	},
	{
		note : 62,
		time : beat * 3.6,
		duration : beat * 0.5
	},
	{
		note : 60,
		time : beat * 4,
		duration : beat * 0.5
	},
	{
		note : 60,
		time : beat * 5,
		duration : beat
	}
]

export class Tutorial extends events.EventEmitter{
	constructor(container){
		super()

		this._tutorial = document.createElement('div')
		this._tutorial.id = 'tutorial'
		container.appendChild(this._tutorial)
	}
	start(){
		if (window.localStorage){
			if (window.localStorage.getItem('showedTutorial') === 'true' && window.location.hash !== '#tutorial'){
				return 
			} else {
				window.localStorage.setItem('showedTutorial', 'true') 
			}
		}

		this._promiseTimeout(400).then(() => {
			this._addText('When you play a few notes', 'user', 4200)
			return this._promiseTimeout(1000)
		}).then(() => {
			const now = Tone.now()
			testMelody.forEach((event) => {
				this.emit('keyDown', event.note, event.time + now)
				this.emit('keyUp', event.note, event.time + event.duration * 0.9 + now)
			})
			return this._promiseTimeout(3000)
		}).then(() => {
			this._sendUserMelody()
			return this._promiseTimeout(500)
		}).then(() => {
			this._addText('the computer will respond to what you play', 'ai', 5000)
		})
	}
	_promiseTimeout(time){
		return new Promise(done => {
			setTimeout(done, time)
		})
	}

	_sendUserMelody(){
		const now = Tone.now()
		testMelody.forEach((event) => {
			this.emit('aiKeyDown', event.note, event.time + now)
			this.emit('aiKeyUp', event.note, event.time + event.duration * 0.9 + now)
		})
	}
	_addText(text, className, time){
		const element = document.createElement('div')
		element.classList.add('text')
		element.classList.add(className)
		element.textContent = text
		this._tutorial.appendChild(element)
		requestAnimationFrame(() => {
			element.classList.add('visible')
		})
		if (time){
			setTimeout(() => {
				this._removeText(element)
			}, time)
		}
		return element
	}

	_removeText(element){
		element.classList.remove('visible')
		setTimeout(() => {
			element.remove()
		}, 500)
	}
}