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

import 'style/about.css'
import events from 'events'

const magentaLink = 'https://magenta.tensorflow.org/'
const tfLink = 'https://www.tensorflow.org/'
const toneLink = 'https://github.com/Tonejs/Tone.js'
const sourceCode = 'https://github.com/tensorflow/magenta/demos/ai-jam-js'
const ogSourceCode = 'https://github.com/googlecreativelab/aiexperiments-ai-duet'

const blurbCopy = `Extended from the <a target='_blank' href='${ogSourceCode}'>
				  AI Duet</a> experiment built by Yotam Mann with
          friends on the Magenta and Creative Lab teams at Google.
					It uses <a target='_blank' href='${tfLink}'>TensorFlow</a>,
					<a target='_blank' href='${toneLink}'>Tone.js</a> and tools
					from the <a target='_blank' href='${magentaLink}'>Magenta project</a>.
					The open-source code is <a target='_blank' href='${sourceCode}'>
					available here</a>.

					<p>Click the keyboard, use your computer keys, or even plug in a MIDI keyboard.</p>

					<p>Keyboard shortcuts:
					<table style="width:75%">
					  <tr><td>Z</td><td><i>Toggles the metronome.</i></td></tr>
					  <tr><td>Q</td><td><i>Toggles between piano and drums.</i></td></tr>
					  <tr><td>LEFT/RIGHT</td><td><i>Cycles through available models.</i></td></tr>
					  <tr><td>UP/DOWN</td><td><i>Adjusts sampling 'temperature'. Higher temperatures sound more random.</i></td></tr>
					  <tr><td>SPACE</td><td><i>Toggles looping of AI sequence.</i></td></tr>
					  <tr><td>M</td><td><i>Mutates AI sequence.</i></td></tr>
					  <tr><td>0-9</td><td><i>Sets AI response duration (in bars). 0 matches your input.</i></td></tr>
					  <tr><td>SHIFT + 0-9</td><td><i>Sets input sequence duration (in bars). 0 matches your input.</i></td></tr>
					  <tr><td>DELETE</td><td><i>Stops current AI playback.</i></td></tr>
					  <tr><td>X</td><td><i>Toggles "solo mode", which stops AI from listening to inputs.</i></td></tr>
					</table></p>
					`

export class About extends events.EventEmitter{
	constructor(container){

		super()

		this._container = document.createElement('div')
		this._container.id = 'about'
		container.appendChild(this._container)

		this._toggleButton = document.createElement('div')
		this._toggleButton.id = 'aboutButton'
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

		const content = document.createElement('div')
		content.id = 'content'
		this._container.appendChild(content)

		const title = document.createElement('div')
		title.id = 'title'
		title.textContent = 'A.I. Jam'
		// content.appendChild(title)

		const blurb = document.createElement('div')
		blurb.id = 'blurb'
		content.appendChild(blurb)
		blurb.innerHTML = blurbCopy

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