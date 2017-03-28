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

		// make the samples loaded based on the screen size
		if (screen.availWidth < 750 && screen.availHeight < 750){
			this._range = [48, 72]
		} else if (screen.availWidth < 1000 && screen.availHeight < 1000){
			this._range = [48, 84]
		} else {
			this._range = [24, 108]			
		}


		this._piano = new Sampler('audio/Salamander/', this._range)

		this._synth = new Sampler('audio/string_ensemble/', this._range)

	}

	load(){
		return Promise.all([this._piano.load(), this._synth.load()])
	}

	keyDown(note, time=Tone.now(), ai=false){
		if (note >= this._range[0] && note <= this._range[1]){
			this._piano.keyDown(note, time)
			if (ai){
				this._synth.volume = -8
				this._synth.keyDown(note, time)
			}
		}


	}

	keyUp(note, time=Tone.now(), ai=false){
		if (note >= this._range[0] && note <= this._range[1]){
			time += 0.05
			this._piano.keyUp(note, time)
			if (ai){
				this._synth.keyUp(note, time)
			}
		}
	}
}

export {Sound}