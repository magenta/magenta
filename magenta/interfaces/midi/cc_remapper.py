# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An interface for user-definable Control Change (CC) messages."""

import mido
import time

class CCRemapper(object):
  """An interface for user-definable Control Change (CC) messages.

  Attributes:
    _inport: The Mido port for receiving messages.
    _cc_map: The dictionary of user defined Control Change messages.

  Args:
    input_midi_port: The string MIDI port name to use for input.
  """

  def __init__(self, input_midi_port, cc_map):
    self._inport = mido.open_input(input_midi_port)
    self._cc_map = cc_map

  def remapper_interface(self):
    """Interface asking for user input of which parameters to remap"""

    while True:
      print "\nWhat would you like to assign a Control Change to...?:"
      print "0. None (Exit)"
      for i, CCparam in enumerate(sorted(self._cc_map.keys())):
        print "{}. {}".format(i + 1, CCparam)
      print ""
      try:
        cc_choice_index = int(raw_input("Please make a selection...\n>>>"))
      except ValueError:
        print "Please enter a number..."
        time.sleep(.5)
        continue
      else:
        if cc_choice_index == 0:
          self._inport.close()
          return self._cc_map
        elif cc_choice_index < 0 or cc_choice_index > len(self._cc_map):
          print("There is no CC Parameter assigned to that "
                "number, please select from the list.")
          time.sleep(.5)
          continue
        else:
          cc_choice = sorted(self._cc_map.keys())[cc_choice_index - 1]

      if cc_choice == 'Start Capture' or cc_choice == 'Stop Capture':
        self.remap_capture_message(self._inport, cc_choice,
                                   self._cc_map['Start Capture'],
                                   self._cc_map['Stop Capture'])
      else:
        self.remap_cc_message(self._inport, cc_choice)

  def remap_cc_message(self, input_port, cc_choice):
    """Defines how to remap control change messages for defined parameters.

    Args:
      input_port: The input port to receive the control change message.
      cc_choice: The CC map dictionary key to assign a message to.
    """

    while True:
      while input_port.receive(block=False) is not None:
        pass
      print ("What control or key would you like to assign to {}?\n"
             "Please press one now...".format(cc_choice))
      msg = input_port.receive()
      if msg.hex().startswith('B'):
        print ("{} has been assigned to controller "
               "number {}".format(cc_choice, msg.control))
        time.sleep(1)
        self._cc_map[cc_choice] = msg
        return
      else:
        print('Sorry, I only accept MIDI CC messages for this parameter..')
        continue

  def remap_capture_message(self, input_port, cc_choice, msg_start_capture,
                            msg_stop_capture):
    """Remap incoming messages for start and stop capture.

    Args:
      input_port: The input port to receive the control change message.
      cc_choice: The CC map dictionary key to assign a message to.
      msg_start_capture: The currently assigned start capture message.
      msg_stop_capture: The currently assigned stop capture message.
    """

    while True:
      while input_port.receive(block=False) is not None:
        pass
      print ("What control or key would you like to assign to {}?\n"
             "Please press one now...".format(cc_choice))
      msg = input_port.receive()
      if msg.type == 'note_on':
        print ("{} has been assigned to a musical note with value {}.\n"
               "This note will not be available for "
               "musical content.".format(cc_choice, msg.note))
        msg = msg.copy(velocity=1)
        break
      elif msg.type == 'control_change':
        print ("{} has been assigned to controller "
               "number {}.".format(cc_choice, msg.control))
        break
      else:
        print ("Sorry, I only accept buttons outputting MIDI CC "
               "and note messages...try again.")
        continue

    if cc_choice == 'Start Capture':
      if msg_stop_capture is None:
        msg_start_capture = msg
      elif (msg.hex()[:5] == msg_stop_capture.hex()[:5] and
            msg.type == 'note_on'):
        print ("Stop Capture is assigned to the same note...\n"
               "This will act as a toggle between Start and Stop.")
        msg_start_capture = msg
      elif msg == msg_stop_capture and msg.type == 'control_change':
        print ("Stop Capture has the same controller number and value...\n"
               "This will act as a toggle between Start and Stop.")
        msg_start_capture = msg
      elif (msg.hex()[:5] == msg_stop_capture.hex()[:5] and
            msg.type =='control_change'):
        print ("Stop Capture has the same controller "
               "number but a different value...\n"
               "A message with max value (127) will start capture.\n"
               "A message with min value (0) will stop capture.")
        msg_start_capture = mido.parse(msg.bytes()[:2] + [127])
        msg_stop_capture = mido.parse(msg.bytes()[:2] + [0])
      else:
        msg_start_capture = msg

    elif cc_choice == 'Stop Capture':
      if msg_start_capture is None:
        msg_stop_capture = msg
      elif (msg.hex()[:5] == msg_start_capture.hex()[:5] and
            msg.type == 'note_on'):
        print ("Start Capture is assigned to the same note...\n"
               "This will act as a toggle between Start and Stop.")
        msg_stop_capture = msg
      elif msg == msg_start_capture and msg.type == 'control_change':
        print ("Start Capture has the same controller number and value...\n"
               "This will act as a toggle between Start and Stop.")
        msg_stop_capture = msg
      elif (msg.hex()[:5] == msg_start_capture.hex()[:5] and
            msg.type == 'control_change'):
        print ("Start Capture has the same controller "
               "number but a different value...\n"
               "A message with max value (127) will start capture.\n"
               "A message with min value (0) will stop capture.")
        msg_start_capture = mido.parse(msg.bytes()[:2] + [127])
        msg_stop_capture = mido.parse(msg.bytes()[:2] + [0])
      else:
        msg_stop_capture = msg

    time.sleep(1)
    self._cc_map['Start Capture'] = msg_start_capture
    self._cc_map['Stop Capture'] = msg_stop_capture
