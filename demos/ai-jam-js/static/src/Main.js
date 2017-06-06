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

import {Keyboard} from 'keyboard/Keyboard'
import {Midi} from 'keyboard/Midi'
import {Sound} from 'sound/Sound'
import {Glow} from 'interface/Glow'
import {Splash} from 'interface/Splash'
import {About} from 'interface/About'
import 'babel-polyfill'

/////////////// SPLASH ///////////////////

const about = new About(document.body)
const splash = new Splash(document.body)

splash.on('click', () => {
	keyboard.activate()
	about.showButton()
})
splash.on('about', () => {
	about.open(true)
})
about.on('close', () => {
	if (!splash.loaded || splash.isOpen()){
		splash.show()
	} else {
		keyboard.activate()
	}
})
about.on('open', () => {
	keyboard.deactivate()
	if (splash.isOpen()){
		splash.hide()
	}
})


/////////////// PIANO ///////////////////

const container = document.createElement('div')
container.id = 'container'
document.body.appendChild(container)


const midi = new Midi()
const glow = new Glow(container)
const keyboard = new Keyboard(container, midi)

const sound = new Sound()
sound.load()

var isShifted = false
document.body.addEventListener('keydown', (e) => {
		if (e.keyCode == 16) {
			isShifted = true
		}
}, true)
document.body.addEventListener('keyup', (e) => {
		if (e.keyCode == 16) {
			isShifted = false
		} else if (isShifted && e.keyCode >= 48 && e.keyCode <= 57) {  // SHIFT + 0-9
      midi.setCallBars(e.keyCode - 48)
		} else if (e.keyCode >= 48 && e.keyCode <= 57) {  // 0-9
      midi.setResponseBars(e.keyCode - 48)
    } else if (e.keyCode == 37) {  // Left arrow
    	midi.adjustBundleIndex(-1)
    } else if (e.keyCode == 39) {  // Right arrow
    	midi.adjustBundleIndex(1)
    } else if (e.keyCode == 32) {  // Space bar
    	midi.toggleLoop()
    } else if (e.keyCode == 77) {  // m
    	midi.triggerMutate()
    } else if (e.keyCode == 38) {  // Up arrow
    	midi.adjustTemperature(2)
    } else if (e.keyCode == 40) {  // Down arrow
    	midi.adjustTemperature(-2)
    } else if (e.keyCode == 8) {  // Backspace/Delete
    	midi.triggerPanic()
    } else if (e.keyCode == 9) {  // Tab
    	keyboard.toggleDrumMode()
		}
}, true)

keyboard.on('keyDown', (note, time, ai=false, drum=false) => {
	sound.keyDown(note, time, ai, drum)
	if (ai) {
		glow.user()
	} {
		glow.ai()
	}
})

keyboard.on('keyUp', (note, time, ai=false, drum=false) => {
	sound.keyUp(note, time, ai, drum)
	if (ai) {
		glow.user()
	} {
		glow.ai()
	}
})