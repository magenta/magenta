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

const THREE = require('three')
import {Roll} from 'roll/Roll'

const geometry = new THREE.PlaneBufferGeometry( 1, 1, 1 )
const material = new THREE.MeshBasicMaterial( {color: 0x1FB7EC, side: THREE.BackSide} )
const aiMaterial = new THREE.MeshBasicMaterial( {color: 0xFFB729, side: THREE.BackSide} )

export class RollNote {
	constructor(element, ai){
		this.element = element
		const box = this.element.getBoundingClientRect()
		const initialScaling = 3000
		this.plane = new THREE.Mesh( geometry, ai ? aiMaterial : material )
		const margin = 4
		const width = box.width - margin * 2
		this.plane.scale.set(width, initialScaling, 1)
		this.plane.position.z = 0
		this.plane.position.x = box.left  + margin + width / 2
		this.plane.position.y = Roll.bottom + initialScaling / 2
		this.bottom = Roll.bottom
		Roll.add(this.plane)
	}
	noteOff(bottom){
		const dist = Roll.bottom - this.bottom
		this.plane.scale.y = Math.max(dist, 5)
		this.plane.position.y = this.bottom + this.plane.scale.y / 2
	}
}