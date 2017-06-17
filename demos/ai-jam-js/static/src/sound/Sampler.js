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

import Frequency from 'Tone/type/Frequency'
import Buffers from 'Tone/core/Buffers'
import MultiPlayer from 'Tone/source/MultiPlayer'
import Tone from 'Tone/core/Tone'
import AudioBuffer from 'Tone/core/Buffer'

class Sampler{
	constructor(baseUrl='', range=[21, 108]){
		this._urls = {}
		for (var note = range[0]; note <= range[1]; note++) {
			this._urls[note] = baseUrl + note + '.mp3'
		}
		this._player = null

		this._loaded = false
		AudioBuffer.on('load', () => {
			this._loaded = true
		})
	}

	load(){
		return new Promise(done => {
			this._player = new MultiPlayer(this._urls, done).toMaster()
			this._player.fadeOut = 0.2
		})
	}

	set volume(vol){
		if (this._loaded){
			this._player.volume.value = vol
		}
	}

	keyDown(note, time){
		if (this._loaded){
			const duration = this._player.buffers.get(note).duration * 0.95
			this._player.start(note, time, 0, duration - this._player.fadeOut)
		}
	}

	keyUp(note, time){
		if (this._loaded){
			this._player.stop(note, time)
		}
	}
}

export {Sampler}