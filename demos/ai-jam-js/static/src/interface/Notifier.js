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

import 'style/notification.css'

export class Notifier {
  constructor(container){
    this._container = document.createElement('div')
    this._container.id = 'notification'
    container.appendChild(this._container)

    const message = document.createElement('div')
    message.id = 'message'
    this._container.appendChild(message)
    this._message = message
  }

  notify(msg){
    this._message.innerHTML = msg

    var that = this
    function removeNotification() {
      if (that._message.innerHTML == msg) {
        that._message.innerHTML = ''
      }
    }
    setTimeout(removeNotification, 2000);
  }

}