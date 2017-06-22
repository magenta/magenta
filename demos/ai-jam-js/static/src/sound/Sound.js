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

import Tone from 'Tone/core/Tone'
import PolySynth from 'Tone/instrument/PolySynth'
import Frequency from 'Tone/type/Frequency'
import MonoSynth from 'Tone/instrument/MonoSynth'
import {Sampler} from 'sound/Sampler'

class Sound {
	constructor(){

		this._range = [24, 108]

		this._piano = new Sampler('audio/piano/', this._range)

		this._e_piano = new Sampler('audio/e_piano/', this._range)

		this._drums = new Sampler('audio/drums/', this._range)

		this._metronome = new Sampler('audio/metronome/', [0, 1])
	}

	load(){
		return Promise.all(
			[this._piano.load(), this._e_piano.load(),
			 this._drums.load(), this._metronome.load()])
	}

	metronomeTick(note) {
		this._metronome.keyDown(note, Tone.now())
		this._metronome.keyUp(note, Tone.now() + 0.15)
	}

	keyDown(note, time=Tone.now(), ai=false, drum=false){
		if (note >= this._range[0] && note <= this._range[1]){
			if (drum) {
			  this._drums.keyDown(note, time)
			} else if (ai){
			  this._e_piano.keyDown(note, time)
			} else {
			  this._piano.keyDown(note, time)
			}
		}
	}

	keyUp(note, time=Tone.now(), ai=false, drum=false){
		if (note >= this._range[0] && note <= this._range[1]){
			time += 0.05
			if (drum) {
			  this._drums.keyUp(note, time)
			} else if (ai){
				this._e_piano.keyUp(note, time)
			} else {
				this._piano.keyUp(note, time)
			}
		}
	}
}

export {Sound}