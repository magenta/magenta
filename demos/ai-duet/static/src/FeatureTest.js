/**
 * Copyright 2017 Google Inc.
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

import 'main.css'
import domReady from 'domready'
import 'splash.css'
import Modernizr from 'exports?Modernizr!Modernizr'

require.ensure(['Main', 'notsupported.css'], (require) => {

	domReady(() => {

		if (Modernizr.webaudio && Modernizr.webgl){

			const main = require('Main')

		} else {

			require('notsupported.css')

			const text = document.createElement('div')
			text.id = 'notsupported'
			text.innerHTML = 'Oops, sorry for the tech trouble. For the best experience, view in <a href="https://www.google.com/chrome" target="_blank">Chrome browser</a>.'
			document.body.appendChild(text)

		}

	})

})


