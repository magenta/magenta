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

import Buffer from 'Tone/core/Buffer'
import 'style/splash.css'
import events from 'events'
import Loader from 'interface/Loader'

class Controls extends events.EventEmitter{
	constructor(container){

		super()

		this._toggleButton = document.createElement('div')
		this._toggleButton.id = 'controlsButton'
		this._toggleButton.classList.add('open')
		container.appendChild(this._toggleButton)
		this._toggleButton.addEventListener('click', (e) => {
			e.preventDefault()
			if (this.isOpen()){
				this.close()
			} else {
				this.open()
			}
		})

		const title = document.createElement('div')
		title.id = 'title'
		title.textContent = 'Controls'
		this._container.appendChild(title)

		const settings = document.createElement('div')
		settings.id = 'settings'
		this._container.appendChild(settings)

		content.appendChild(title)

		const loop = document.createElement('div')
		loop.id = 'loopButton'
		content.appendChild(loop)
		loop.textContent = 'Loop'
		loop.classList.add('setting')
		settings.addEventListener('click', () => {
			this.emit('loop')

	}

	close(){
		this._toggleButton.classList.remove('close')
		this._toggleButton.classList.add('open')

		this._container.classList.remove('visible')

		this.emit('close')
		if (window.ga){
			ga('send', 'event', 'AI-Duet', 'Click', 'About - Close')
		}
	}
	open(){
		this._toggleButton.classList.add('close')
		this._toggleButton.classList.remove('open')

		this._playButton.classList.add('visible')
		this._container.classList.add('visible')
		this.emit('open')
		if (window.ga){
			ga('send', 'event', 'AI-Duet', 'Click', 'About - Open')
		}
	}

	isOpen(){
		return this._container.classList.contains('visible')
	}

	showButton(){
		this._toggleButton.classList.add('show')
	}
}