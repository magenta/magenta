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

import 'style/glow.css'
import Tone from 'Tone/core/Tone'

class Glow {
	constructor(container){
		this._element = document.createElement('div')
		this._element.id = 'glow'
		container.appendChild(this._element)

		this._aiGlow = document.createElement('div')
		this._aiGlow.id = 'ai'
		this._element.appendChild(this._aiGlow)

		this._userGlow = document.createElement('div')
		this._userGlow.id = 'user'
		this._element.appendChild(this._userGlow)

		this._aiTime = -1

		this._boundLoop = this._loop.bind(this)
		this._loop()

		this._aiVisible = true
	}
	_loop(){
		requestAnimationFrame(this._boundLoop)
		if (this._aiTime < 0 || Tone.now() > this._aiTime){
			if (this._aiVisible){
				this._aiVisible = false
				this._aiGlow.classList.remove('visible')
				this._userGlow.classList.add('visible')
			}
		} else {
			if (!this._aiVisible){
				this._aiVisible = true
				this._aiGlow.classList.add('visible')
				this._userGlow.classList.remove('visible')
			}
		}
	}
	ai(time){
		this._aiTime = Math.max(this._aiTime, time + 0.3)
	}
	user(){
		this._aiTime = -1
	}
}

export {Glow}