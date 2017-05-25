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

import {RollNote} from 'roll/RollNote'

export class Note{
	constructor(container, ai){
		this.element = document.createElement('div')
		this.element.classList.add('highlight')
		this.element.classList.add('active')
		if (ai){
			this.element.classList.add('ai')
		}
		container.appendChild(this.element)

		this.rollNote = new RollNote(container, ai)
	}
	noteOff(){
		this.element.classList.remove('active')
		this.rollNote.noteOff()
		setTimeout(() => {
			this.element.remove()
		}, 1000)
	}
}