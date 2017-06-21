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

import events from 'events'
import 'style/controls.css'

class Controls extends events.EventEmitter {

  constructor(container, midi, magenta, keyboard){
    super()
    this._container = document.createElement('div')
    this._container.id = 'controls'
    container.appendChild(this._container)

    this._midi = midi
    this._magenta = magenta
    this._keyboard = keyboard

    // Piano and Drums
    this._addTitle('Instrument:')
    this._addDivider(37)
    this._pianoButton = this._addButton(
        'Piano', this.toggleInstrument.bind(this))
    this._pianoButton.classList.add('active')
    this._drumButton = this._addButton(
        'Drums', this.toggleInstrument.bind(this))
    this._addDivider()

    // Metronome
    this._metronomeButton = this._addButton(
        'Metronome', this.toggleMetronome.bind(this))
    this._container.appendChild(document.createElement('br'))

    // CALL
    // Call Bars
    this._addTitle('Call Bars:')
    this._addDivider(50)
    this._callBarButtons = []
    this._callBarButtons.push(
      this._addButton('Auto', this.setCallBars.bind(this, 0)))
    this._callBarButtons[0].classList.add('active')
    for (var i = 1; i <= 9; i++) {
      this._callBarButtons.push(
          this._addButton(i, this.setCallBars.bind(this, i)))
    }
    this._addDivider()
    // Solo
    this._soloButton = this._addButton('Solo', this.toggleSolo.bind(this))
    this._container.appendChild(document.createElement('br'))


    // RESPONSE
    // Response Bars
    this._addTitle('Response Bars:')
    this._addDivider(8)
    this._responseBarButtons = []
    this._responseBarButtons.push(
      this._addButton('Auto', this.setResponseBars.bind(this, 0)))
    this._responseBarButtons[0].classList.add('active')
    for (var i = 1; i <= 9; i++) {
      this._responseBarButtons.push(
          this._addButton(i, this.setResponseBars.bind(this, i)))
    }
    this._addDivider()
    // Loop
    this._loopButton = this._addButton('Loop', this.toggleLoop.bind(this))
    this._addDivider()
    // Mutate
    this._mutateButton = this._addButton(
        'Mutate', this.triggerMutate.bind(this))
    this._addDivider()
    // Clear
    this._panicButton = this._addButton('Clear', this.triggerPanic.bind(this))
    this._addDivider()
    this._container.appendChild(document.createElement('br'))

    // Model
    this._addTitle('Model:')
    this._addDivider(69)
    this._modelButtons = []
    for (var i = 0; i < this._magenta.instances().length; ++i) {
      var inst = this._magenta.instance(i)
      var instModelButtons = []
      for (var j = 0; j < inst.models().length; ++j) {
        var button = this._addButton(
            inst.models()[j], this.setModel.bind(this, j))
        if (i != 0) {
          button.style.visibility = 'hidden'
        }
        if (j == 0) {
          button.classList.add('active')
        }
        instModelButtons.push(button)
      }
      this._modelButtons.push(instModelButtons)
    }
    this._addDivider()
  }

  _addTitle(text) {
    var title = document.createElement('div')
    title.classList.add('title')
    title.textContent = text
    this._container.appendChild(title)
  }

  _addDivider(w=15) {
    var divider = document.createElement('div')
    divider.classList.add('divider')
    divider.setAttribute('style', 'width:' +  w + 'px')
    this._container.appendChild(divider)
  }

  _addButton(text, callback) {
    var button = document.createElement('button')
    button.textContent = text
    button.addEventListener('pointerup', callback)
    this._container.appendChild(button)
    return button
  }

  _toggle(button) {
    if (button.classList.contains('active')) {
      button.classList.remove('active')
    } else {
      button.classList.add('active')
    }
  }

  _trigger(button) {
    button.classList.add('active')
    var that = this
    setTimeout(() => {button.classList.remove('active')}, 500)
  }

  toggleMetronome() {
    this._toggle(this._metronomeButton)
    this._midi.toggleMetronome()
  }

  toggleLoop() {
    this._toggle(this._loopButton)
    this._magenta.selected().toggleLoop()
  }

  toggleSolo() {
    this._toggle(this._soloButton)
    this._keyboard.toggleSoloMode()
  }

  toggleInstrument() {
    if (this._pianoButton.classList.contains('active')) {
      this._pianoButton.classList.remove('active')
      this._drumButton.classList.add('active')
    } else {
      this._pianoButton.classList.add('active')
      this._drumButton.classList.remove('active')
    }

    this._modelButtons[this._magenta.selectedIndex()].forEach((button) => {
      button.style.visibility = 'hidden'
    })
    this._magenta.toggleSelected()
    this._modelButtons[this._magenta.selectedIndex()].forEach((button) => {
      button.style.visibility = 'visible'
    })

    this._keyboard.toggleDrumMode()

    // Reset button states.
    this.setCallBars(this._magenta.selected().callBars())
    this.setResponseBars(this._magenta.selected().responseBars())
    if (this._magenta.selected().isLooping()) {
      this._loopButton.classList.add('active')
    } else {
      this._loopButton.classList.remove('active')
    }

  }

  setCallBars(numBars) {
    this._callBarButtons.forEach((button) => {
      button.classList.remove('active')})
    this._callBarButtons[numBars].classList.add('active')
    this._magenta.selected().setCallBars(numBars)

  }

  setResponseBars(numBars) {
    this._responseBarButtons.forEach((button) => {
      button.classList.remove('active')})
    this._responseBarButtons[numBars].classList.add('active')
    this._magenta.selected().setResponseBars(numBars)
  }


  triggerMutate() {
    this._trigger(this._mutateButton)
    this._magenta.selected().triggerMutate()
  }

  triggerPanic() {
    this._trigger(this._panicButton)
    this._magenta.selected().triggerPanic()
  }

  adjustModelIndex(amt) {
    var j = this._magenta.selected().bundleIndex() + amt
    if (j >= this._magenta.selected().models().length) {
      j = 0
    } else if (j < 0) {
      j = this._magenta.selected().models().length - 1
    }
    this.setModel(j)
  }

  setModel(j) {
    var i = this._magenta.selectedIndex()
    this._modelButtons[i].forEach((button) => {
      button.classList.remove('active')})
    this._modelButtons[i][j].classList.add('active')
    this._magenta.selected().setBundleIndex(j)
  }

}

export {Controls}