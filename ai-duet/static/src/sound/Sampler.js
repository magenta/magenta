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

		//all the notes of the piano sampled every 3rd note
		const notes = [21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108]

		const lowerIndex = notes.findIndex((note) => note >= range[0])
		let upperIndex = notes.findIndex((note) => note >= range[1])
		upperIndex = upperIndex === -1 ? upperIndex = notes.length : upperIndex + 1

		const slicedNotes = notes.slice(lowerIndex, upperIndex)

		this._urls = {}
		slicedNotes.forEach(note => {
			this._urls[note - 1] = baseUrl + Frequency(note, 'midi').toNote().replace('#', 's') + '.mp3'
			this._urls[note] = baseUrl + Frequency(note, 'midi').toNote().replace('#', 's') + '.mp3'
			this._urls[note + 1] = baseUrl + Frequency(note, 'midi').toNote().replace('#', 's') + '.mp3'
		})
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
			let pitch = this._midiToFrequencyPitch(note)
			const duration = this._player.buffers.get(note).duration * 0.95
			this._player.start(note, time, 0, duration - this._player.fadeOut, pitch)
		}
	}

	keyUp(note, time){
		if (this._loaded){
			this._player.stop(note, time)
		}
	}

	_midiToFrequencyPitch(midi){
		let mod = midi % 3
		if (mod === 1){
			return 1
		} else if (mod === 2){
			return -1
		} else {
			return 0
		}
	}
}

export {Sampler}