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

const geometry = new THREE.PlaneGeometry( 1, 1, 1 )
const material = new THREE.MeshBasicMaterial( {color: 0x1FB7EC, side: THREE.DoubleSide} )
const aiMaterial = new THREE.MeshBasicMaterial( {color: 0xFFB729, side: THREE.DoubleSide} )

window.zero = new THREE.Vector3(0, 0, 0)

function scale(value, inMin, inMax, min, max){
	return ((value - inMin) / (inMax - inMin)) * (max - min) + min
}

class RollClass {
	constructor(container){
		this._element = document.createElement('div')
		this._element.id = 'roll'

		this._camera = new THREE.OrthographicCamera(0, 1, 1, 0, 1, 1000 )
		this._camera.position.z = 1
		this._camera.lookAt(new THREE.Vector3(0, 0, 0))

		this._scene = new THREE.Scene()

		this._renderer = new THREE.WebGLRenderer({alpha: true})
		this._renderer.setClearColor(0x000000, 0)
		this._renderer.setPixelRatio( window.devicePixelRatio )
		this._renderer.sortObjects = false
		this._element.appendChild(this._renderer.domElement)

		this._currentNotes = {}

		window.camera = this._camera

		//start the loop
		this._lastUpdate = Date.now()
		this._boundLoop = this._loop.bind(this)
		this._boundLoop()
		window.addEventListener('resize', this._resize.bind(this))
	}

	get bottom(){
		return this._element.clientHeight + this._camera.position.y
	}

	appendTo(container){
		container.appendChild(this._element)
		this._resize()
	}

	add(element){
		this._scene.add(element)
	}

	keyDown(midi, box, ai=false){
		const selector = ai ? `ai${midi}` : midi
		if (!this._currentNotes.hasOwnProperty(selector)){
			this._currentNotes[selector] = []
		}
		if (midi && box){
			//translate the box coords to this space
			const initialScaling = 10000
			const plane = new THREE.Mesh( geometry, ai ? aiMaterial : material )
			const margin = 4
			const width = box.width - margin * 2
			plane.scale.set(width, initialScaling, 1)
			plane.position.z = 0
			plane.position.x = box.left  + margin + width / 2
			plane.position.y = this._element.clientHeight + this._camera.position.y + initialScaling / 2
			this._scene.add(plane)

			this._currentNotes[selector].push({
				plane : plane,
				position: this._camera.position.y
			})
		}

	}

	keyUp(midi, ai=false){
		const selector = ai ? `ai${midi}` : midi
		if (this._currentNotes[selector] && this._currentNotes[selector].length){
			const note = this._currentNotes[selector].shift()
			const plane = note.plane
			const position = note.position
			// get the distance covered
			plane.scale.y = Math.max(this._camera.position.y - position, 5)
			plane.position.y = this._element.clientHeight + position + plane.scale.y / 2
		}
	}

	_resize(){
		const frustumSize = 1000
		const aspect = this._element.clientWidth / this._element.clientHeight
		//make it match the screen pixesl
		this._camera.left 	=	0
		this._camera.bottom	=	this._element.clientHeight
		this._camera.right  =   this._element.clientWidth
		this._camera.top    =   0

		//update things
		this._camera.updateProjectionMatrix()
		this._renderer.setSize( this._element.clientWidth, this._element.clientHeight )
	}

	_loop(){
		const delta = Date.now() - this._lastUpdate
		this._lastUpdate = Date.now()
		requestAnimationFrame(this._boundLoop)
		this._renderer.render( this._scene, this._camera )
		this._camera.position.y += 1 / 10 * delta
	}
}

const Roll = new RollClass()
export {Roll}