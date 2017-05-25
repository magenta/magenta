function AudioKeys(options) {
  var self = this;

  self._setState(options);

  // all listeners are stored in arrays in their respective properties.
  // e.g. self._listeners.down = [fn1, fn2, ... ]
  self._listeners = {};

  // bind DOM events
  self._bind();
}

// Play well with require so that we can run a test suite and use browserify.
if(typeof module !== 'undefined') {
  module.exports = AudioKeys;
}

AudioKeys.prototype._setState = function(options) {
  var self = this;

  if(!options) {
    options = {};
  }

  // the state is kept in this object
  self._state = {};

  // set some defaults ...
  self._extendState({
    polyphony: 4,
    rows: 1,
    priority: 'last',
    rootNote: 60,
    octaveControls: true,
    octave: 0,
    velocityControls: true,
    velocity: 127,
    keys: [],
    buffer: []
  });

  // ... and override them with options.
  self._extendState(options);
};

AudioKeys.prototype._extendState = function(options) {
  var self = this;

  for(var o in options) {
    self._state[o] = options[o];
  }
};

AudioKeys.prototype.set = function(/* options || property, value */) {
  var self = this;

  if(arguments.length === 1) {
    self._extendState(arguments[0]);
  } else {
    self._state[arguments[0]] = arguments[1];
  }

  return this;
};

AudioKeys.prototype.get = function(property) {
  var self = this;

  return self._state[property];
};

// ================================================================
// Event Listeners
// ================================================================

// AudioKeys has a very simple event handling system. Internally
// we'll call self._trigger('down', argument) when we want to fire
// an event for the user.

AudioKeys.prototype.down = function(fn) {
  var self = this;

  // add the function to our list of listeners
  self._listeners.down = (self._listeners.down || []).concat(fn);
};

AudioKeys.prototype.up = function(fn) {
  var self = this;

  // add the function to our list of listeners
  self._listeners.up = (self._listeners.up || []).concat(fn);
};

AudioKeys.prototype._trigger = function(action /* args */) {
  var self = this;

  // if we have any listeners by this name ...
  if(self._listeners[action] && self._listeners[action].length) {
    // grab the arguments to pass to the listeners ...
    var args = Array.prototype.slice.call(arguments);
    args.splice(0, 1);
    // and call them!
    self._listeners[action].forEach( function(fn) {
      fn.apply(self, args);
    });
  }
};

// ================================================================
// DOM Bindings
// ================================================================

AudioKeys.prototype._bind = function() {
  var self = this;

  if(typeof window !== 'undefined' && window.document) {
    window.document.addEventListener('keydown', function(e) {
      self._addKey(e);
    });
    window.document.addEventListener('keyup', function(e) {
      self._removeKey(e);
    });

    var lastFocus = true;
    setInterval( function() {
      if(window.document.hasFocus() === lastFocus) {
        return;
      }
      lastFocus = !lastFocus;
      if(!lastFocus) {
        self.clear();
      }
    }, 100);
  }
};

// _map returns the midi note for a given keyCode.
AudioKeys.prototype._map = function(keyCode) {
  return this._keyMap[this._state.rows][keyCode] + this._offset();
};

AudioKeys.prototype._offset = function() {
  return this._state.rootNote - this._keyMap[this._state.rows].root + (this._state.octave * 12);
};

// _isNote determines whether a keyCode is a note or not.
AudioKeys.prototype._isNote = function(keyCode) {
  return !!this._keyMap[this._state.rows][keyCode];
};

// convert a midi note to a frequency. we assume here that _map has
// already been called (to account for a potential rootNote offset)
AudioKeys.prototype._toFrequency = function(note) {
  return ( Math.pow(2, ( note-69 ) / 12) ) * 440.0;
};

// the object keys correspond to `rows`, so `_keyMap[rows]` should
// retrieve that particular mapping.
AudioKeys.prototype._keyMap = {
  1: {
    root: 60,
    // starting with the 'a' key
    65:  60,
    87:  61,
    83:  62,
    69:  63,
    68:  64,
    70:  65,
    84:  66,
    71:  67,
    89:  68,
    72:  69,
    85:  70,
    74:  71,
    75:  72,
    79:  73,
    76:  74,
    80:  75,
    186: 76,
    222: 77
  },
  2: {
    root: 60,
    // bottom row
    90:  60,
    83:  61,
    88:  62,
    68:  63,
    67:  64,
    86:  65,
    71:  66,
    66:  67,
    72:  68,
    78:  69,
    74:  70,
    77:  71,
    188: 72,
    76:  73,
    190: 74,
    186: 75,
    191: 76,
    // top row
    81:  72,
    50:  73,
    87:  74,
    51:  75,
    69:  76,
    82:  77,
    53:  78,
    84:  79,
    54:  80,
    89:  81,
    55:  82,
    85:  83,
    73:  84,
    57:  85,
    79:  86,
    48:  87,
    80:  88,
    219: 89,
    187: 90,
    221: 91
  }
};

// ================================================================
// KEY BUFFER
// ================================================================

// The process is:

// key press
//   add to self._state.keys
//   (an accurate representation of keys currently pressed)
// resolve self.buffer
//   based on polyphony and priority, determine the notes
//   that get triggered for the user

AudioKeys.prototype._addKey = function(e) {
  var self = this;
  // if the keyCode is one that can be mapped and isn't
  // already pressed, add it to the key object.
  if(self._isNote(e.keyCode) && !self._isPressed(e.keyCode)) {
    var newKey = self._makeNote(e.keyCode);
    // add the newKey to the list of keys
    self._state.keys = (self._state.keys || []).concat(newKey);
    // reevaluate the active notes based on our priority rules.
    // give it the new note to use if there is an event to trigger.
    self._update();
  } else if(self._isSpecialKey(e.keyCode)) {
    self._specialKey(e.keyCode);
  }
};

AudioKeys.prototype._removeKey = function(e) {
  var self = this;
  // if the keyCode is active, remove it from the key object.
  if(self._isPressed(e.keyCode)) {
    var keyToRemove;
    for(var i = 0; i < self._state.keys.length; i++) {
      if(self._state.keys[i].keyCode === e.keyCode) {
        keyToRemove = self._state.keys[i];
        break;
      }
    }

    // remove the key from _keys
    self._state.keys.splice(self._state.keys.indexOf(keyToRemove), 1);
    self._update();
  }
};

AudioKeys.prototype._isPressed = function(keyCode) {
  var self = this;

  if(!self._state.keys || !self._state.keys.length) {
    return false;
  }

  for(var i = 0; i < self._state.keys.length; i++) {
    if(self._state.keys[i].keyCode === keyCode) {
      return true;
    }
  }
  return false;
};

// turn a key object into a note object for the event listeners.
AudioKeys.prototype._makeNote = function(keyCode) {
  var self = this;
  return {
    keyCode: keyCode,
    note: self._map(keyCode),
    frequency: self._toFrequency( self._map(keyCode) ),
    velocity: self._state.velocity
  };
};

// clear any active notes
AudioKeys.prototype.clear = function() {
  var self = this;
  // trigger note off for the notes in the buffer before
  // removing them.
  self._state.buffer.forEach( function(key) {
    self._trigger('up', key);
  });
  self._state.keys = [];
  self._state.buffer = [];
};

// ================================================================
// NOTE BUFFER
// ================================================================

// every time a change is made to _keys due to a key on or key off
// we need to call `_update`. It compares the `_keys` array to the
// `buffer` array, which is the array of notes that are really
// being played, makes the necessary changes to `buffer` and
// triggers any events that need triggering.

AudioKeys.prototype._update = function() {
  var self = this;

  // a key has been added to self._state.keys.
  // stash the old buffer
  var oldBuffer = self._state.buffer;
  // set the new priority in self.state._keys
  self._prioritize();
  // compare the buffers and trigger events based on
  // the differences.
  self._diff(oldBuffer);
};

AudioKeys.prototype._diff = function(oldBuffer) {
  var self = this;

  // if it's not in the OLD buffer, it's a note ON.
  // if it's not in the NEW buffer, it's a note OFF.

  var oldNotes = oldBuffer.map( function(key) {
    return key.keyCode;
  });

  var newNotes = self._state.buffer.map( function(key) {
    return key.keyCode;
  });

  // check for old (removed) notes
  var notesToRemove = [];
  oldNotes.forEach( function(key) {
    if(newNotes.indexOf(key) === -1) {
      notesToRemove.push(key);
    }
  });

  // check for new notes
  var notesToAdd = [];
  newNotes.forEach( function(key) {
    if(oldNotes.indexOf(key) === -1) {
      notesToAdd.push(key);
    }
  });

  notesToAdd.forEach( function(key) {
    for(var i = 0; i < self._state.buffer.length; i++) {
      if(self._state.buffer[i].keyCode === key) {
        self._trigger('down', self._state.buffer[i]);
        break;
      }
    }
  });

  notesToRemove.forEach( function(key) {
    // these need to fire the entire object
    for(var i = 0; i < oldBuffer.length; i++) {
      if(oldBuffer[i].keyCode === key) {
        self._trigger('up', oldBuffer[i]);
        break;
      }
    }
  });
};

AudioKeys.prototype._prioritize = function() {
  var self = this;

  // if all the keys have been turned off, no need
  // to do anything here.
  if(!self._state.keys.length) {
    self._state.buffer = [];
    return;
  }


  if(self._state.polyphony >= self._state.keys.length) {
    // every note is active
    self._state.keys = self._state.keys.map( function(key) {
      key.isActive = true;
      return key;
    });
  } else {
    // set all keys to inactive.
    self._state.keys = self._state.keys.map( function(key) {
      key.isActive = false;
      return key;
    });

    self['_' + self._state.priority]();
  }

  // now take the isActive keys and set the new buffer.
  self._state.buffer = [];

  self._state.keys.forEach( function(key) {
    if(key.isActive) {
      self._state.buffer.push(key);
    }
  });

  // done.
};

AudioKeys.prototype._last = function() {
  var self = this;
  // set the last bunch to active based on the polyphony.
  for(var i = self._state.keys.length - self._state.polyphony; i < self._state.keys.length; i++) {
    self._state.keys[i].isActive = true;
  }
};

AudioKeys.prototype._first = function() {
  var self = this;
  // set the last bunch to active based on the polyphony.
  for(var i = 0; i < self._state.polyphony; i++) {
    self._state.keys[i].isActive = true;
  }
};

AudioKeys.prototype._highest = function() {
  var self = this;
  // get the highest notes and set them to active
  var notes = self._state.keys.map( function(key) {
    return key.note;
  });

  notes.sort( function(b,a) {
    if(a === b) {
      return 0;
    }
    return a < b ? -1 : 1;
  });

  notes.splice(self._state.polyphony, Number.MAX_VALUE);

  self._state.keys.forEach( function(key) {
    if(notes.indexOf(key.note) !== -1) {
      key.isActive = true;
    }
  });
};

AudioKeys.prototype._lowest = function() {
  var self = this;
  // get the lowest notes and set them to active
  var notes = self._state.keys.map( function(key) {
    return key.note;
  });

  notes.sort( function(a,b) {
    if(a === b) {
      return 0;
    }
    return a < b ? -1 : 1;
  });

  notes.splice(self._state.polyphony, Number.MAX_VALUE);

  self._state.keys.forEach( function(key) {
    if(notes.indexOf(key.note) !== -1) {
      key.isActive = true;
    }
  });
};

// This file maps special keys to the state— octave shifting and
// velocity selection, both available when `rows` = 1.

AudioKeys.prototype._isSpecialKey = function(keyCode) {
  return (this._state.rows === 1 && this._specialKeyMap[keyCode]);
};

AudioKeys.prototype._specialKey = function(keyCode) {
  var self = this;
  if(self._specialKeyMap[keyCode].type === 'octave' && self._state.octaveControls) {
    // shift the state of the `octave`
    self._state.octave += self._specialKeyMap[keyCode].value;
  } else if(self._specialKeyMap[keyCode].type === 'velocity' && self._state.velocityControls) {
    // set the `velocity` to a new value
    self._state.velocity = self._specialKeyMap[keyCode].value;
  }
};

AudioKeys.prototype._specialKeyMap = {
  // octaves
  90: {
    type: 'octave',
    value: -1
  },
  88: {
    type: 'octave',
    value: 1
  },
  // velocity
  49: {
    type: 'velocity',
    value: 1
  },
  50: {
    type: 'velocity',
    value: 14
  },
  51: {
    type: 'velocity',
    value: 28
  },
  52: {
    type: 'velocity',
    value: 42
  },
  53: {
    type: 'velocity',
    value: 56
  },
  54: {
    type: 'velocity',
    value: 70
  },
  55: {
    type: 'velocity',
    value: 84
  },
  56: {
    type: 'velocity',
    value: 98
  },
  57: {
    type: 'velocity',
    value: 112
  },
  48: {
    type: 'velocity',
    value: 127
  },
};

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIkF1ZGlvS2V5cy5qcyIsIkF1ZGlvS2V5cy5zdGF0ZS5qcyIsIkF1ZGlvS2V5cy5ldmVudHMuanMiLCJBdWRpb0tleXMubWFwcGluZy5qcyIsIkF1ZGlvS2V5cy5idWZmZXIuanMiLCJBdWRpb0tleXMucHJpb3JpdHkuanMiLCJBdWRpb0tleXMuc3BlY2lhbC5qcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUNqQkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FDckRBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUNoRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ3hGQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FDN0pBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUNwR0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSIsImZpbGUiOiJhdWRpb2tleXMuanMiLCJzb3VyY2VzQ29udGVudCI6WyJmdW5jdGlvbiBBdWRpb0tleXMob3B0aW9ucykge1xuICB2YXIgc2VsZiA9IHRoaXM7XG5cbiAgc2VsZi5fc2V0U3RhdGUob3B0aW9ucyk7XG5cbiAgLy8gYWxsIGxpc3RlbmVycyBhcmUgc3RvcmVkIGluIGFycmF5cyBpbiB0aGVpciByZXNwZWN0aXZlIHByb3BlcnRpZXMuXG4gIC8vIGUuZy4gc2VsZi5fbGlzdGVuZXJzLmRvd24gPSBbZm4xLCBmbjIsIC4uLiBdXG4gIHNlbGYuX2xpc3RlbmVycyA9IHt9O1xuXG4gIC8vIGJpbmQgRE9NIGV2ZW50c1xuICBzZWxmLl9iaW5kKCk7XG59XG5cbi8vIFBsYXkgd2VsbCB3aXRoIHJlcXVpcmUgc28gdGhhdCB3ZSBjYW4gcnVuIGEgdGVzdCBzdWl0ZSBhbmQgdXNlIGJyb3dzZXJpZnkuXG5pZih0eXBlb2YgbW9kdWxlICE9PSAndW5kZWZpbmVkJykge1xuICBtb2R1bGUuZXhwb3J0cyA9IEF1ZGlvS2V5cztcbn1cbiIsIkF1ZGlvS2V5cy5wcm90b3R5cGUuX3NldFN0YXRlID0gZnVuY3Rpb24ob3B0aW9ucykge1xuICB2YXIgc2VsZiA9IHRoaXM7XG5cbiAgaWYoIW9wdGlvbnMpIHtcbiAgICBvcHRpb25zID0ge307XG4gIH1cblxuICAvLyB0aGUgc3RhdGUgaXMga2VwdCBpbiB0aGlzIG9iamVjdFxuICBzZWxmLl9zdGF0ZSA9IHt9O1xuXG4gIC8vIHNldCBzb21lIGRlZmF1bHRzIC4uLlxuICBzZWxmLl9leHRlbmRTdGF0ZSh7XG4gICAgcG9seXBob255OiA0LFxuICAgIHJvd3M6IDEsXG4gICAgcHJpb3JpdHk6ICdsYXN0JyxcbiAgICByb290Tm90ZTogNjAsXG4gICAgb2N0YXZlQ29udHJvbHM6IHRydWUsXG4gICAgb2N0YXZlOiAwLFxuICAgIHZlbG9jaXR5Q29udHJvbHM6IHRydWUsXG4gICAgdmVsb2NpdHk6IDEyNyxcbiAgICBrZXlzOiBbXSxcbiAgICBidWZmZXI6IFtdXG4gIH0pO1xuXG4gIC8vIC4uLiBhbmQgb3ZlcnJpZGUgdGhlbSB3aXRoIG9wdGlvbnMuXG4gIHNlbGYuX2V4dGVuZFN0YXRlKG9wdGlvbnMpO1xufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5fZXh0ZW5kU3RhdGUgPSBmdW5jdGlvbihvcHRpb25zKSB7XG4gIHZhciBzZWxmID0gdGhpcztcblxuICBmb3IodmFyIG8gaW4gb3B0aW9ucykge1xuICAgIHNlbGYuX3N0YXRlW29dID0gb3B0aW9uc1tvXTtcbiAgfVxufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5zZXQgPSBmdW5jdGlvbigvKiBvcHRpb25zIHx8IHByb3BlcnR5LCB2YWx1ZSAqLykge1xuICB2YXIgc2VsZiA9IHRoaXM7XG5cbiAgaWYoYXJndW1lbnRzLmxlbmd0aCA9PT0gMSkge1xuICAgIHNlbGYuX2V4dGVuZFN0YXRlKGFyZ3VtZW50c1swXSk7XG4gIH0gZWxzZSB7XG4gICAgc2VsZi5fc3RhdGVbYXJndW1lbnRzWzBdXSA9IGFyZ3VtZW50c1sxXTtcbiAgfVxuXG4gIHJldHVybiB0aGlzO1xufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5nZXQgPSBmdW5jdGlvbihwcm9wZXJ0eSkge1xuICB2YXIgc2VsZiA9IHRoaXM7XG5cbiAgcmV0dXJuIHNlbGYuX3N0YXRlW3Byb3BlcnR5XTtcbn07XG4iLCIvLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4vLyBFdmVudCBMaXN0ZW5lcnNcbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cblxuLy8gQXVkaW9LZXlzIGhhcyBhIHZlcnkgc2ltcGxlIGV2ZW50IGhhbmRsaW5nIHN5c3RlbS4gSW50ZXJuYWxseVxuLy8gd2UnbGwgY2FsbCBzZWxmLl90cmlnZ2VyKCdkb3duJywgYXJndW1lbnQpIHdoZW4gd2Ugd2FudCB0byBmaXJlXG4vLyBhbiBldmVudCBmb3IgdGhlIHVzZXIuXG5cbkF1ZGlvS2V5cy5wcm90b3R5cGUuZG93biA9IGZ1bmN0aW9uKGZuKSB7XG4gIHZhciBzZWxmID0gdGhpcztcblxuICAvLyBhZGQgdGhlIGZ1bmN0aW9uIHRvIG91ciBsaXN0IG9mIGxpc3RlbmVyc1xuICBzZWxmLl9saXN0ZW5lcnMuZG93biA9IChzZWxmLl9saXN0ZW5lcnMuZG93biB8fCBbXSkuY29uY2F0KGZuKTtcbn07XG5cbkF1ZGlvS2V5cy5wcm90b3R5cGUudXAgPSBmdW5jdGlvbihmbikge1xuICB2YXIgc2VsZiA9IHRoaXM7XG5cbiAgLy8gYWRkIHRoZSBmdW5jdGlvbiB0byBvdXIgbGlzdCBvZiBsaXN0ZW5lcnNcbiAgc2VsZi5fbGlzdGVuZXJzLnVwID0gKHNlbGYuX2xpc3RlbmVycy51cCB8fCBbXSkuY29uY2F0KGZuKTtcbn07XG5cbkF1ZGlvS2V5cy5wcm90b3R5cGUuX3RyaWdnZXIgPSBmdW5jdGlvbihhY3Rpb24gLyogYXJncyAqLykge1xuICB2YXIgc2VsZiA9IHRoaXM7XG5cbiAgLy8gaWYgd2UgaGF2ZSBhbnkgbGlzdGVuZXJzIGJ5IHRoaXMgbmFtZSAuLi5cbiAgaWYoc2VsZi5fbGlzdGVuZXJzW2FjdGlvbl0gJiYgc2VsZi5fbGlzdGVuZXJzW2FjdGlvbl0ubGVuZ3RoKSB7XG4gICAgLy8gZ3JhYiB0aGUgYXJndW1lbnRzIHRvIHBhc3MgdG8gdGhlIGxpc3RlbmVycyAuLi5cbiAgICB2YXIgYXJncyA9IEFycmF5LnByb3RvdHlwZS5zbGljZS5jYWxsKGFyZ3VtZW50cyk7XG4gICAgYXJncy5zcGxpY2UoMCwgMSk7XG4gICAgLy8gYW5kIGNhbGwgdGhlbSFcbiAgICBzZWxmLl9saXN0ZW5lcnNbYWN0aW9uXS5mb3JFYWNoKCBmdW5jdGlvbihmbikge1xuICAgICAgZm4uYXBwbHkoc2VsZiwgYXJncyk7XG4gICAgfSk7XG4gIH1cbn07XG5cbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbi8vIERPTSBCaW5kaW5nc1xuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG5BdWRpb0tleXMucHJvdG90eXBlLl9iaW5kID0gZnVuY3Rpb24oKSB7XG4gIHZhciBzZWxmID0gdGhpcztcblxuICBpZih0eXBlb2Ygd2luZG93ICE9PSAndW5kZWZpbmVkJyAmJiB3aW5kb3cuZG9jdW1lbnQpIHtcbiAgICB3aW5kb3cuZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsIGZ1bmN0aW9uKGUpIHtcbiAgICAgIHNlbGYuX2FkZEtleShlKTtcbiAgICB9KTtcbiAgICB3aW5kb3cuZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcigna2V5dXAnLCBmdW5jdGlvbihlKSB7XG4gICAgICBzZWxmLl9yZW1vdmVLZXkoZSk7XG4gICAgfSk7XG5cbiAgICB2YXIgbGFzdEZvY3VzID0gdHJ1ZTtcbiAgICBzZXRJbnRlcnZhbCggZnVuY3Rpb24oKSB7XG4gICAgICBpZih3aW5kb3cuZG9jdW1lbnQuaGFzRm9jdXMoKSA9PT0gbGFzdEZvY3VzKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGxhc3RGb2N1cyA9ICFsYXN0Rm9jdXM7XG4gICAgICBpZighbGFzdEZvY3VzKSB7XG4gICAgICAgIHNlbGYuY2xlYXIoKTtcbiAgICAgIH1cbiAgICB9LCAxMDApO1xuICB9XG59O1xuIiwiLy8gX21hcCByZXR1cm5zIHRoZSBtaWRpIG5vdGUgZm9yIGEgZ2l2ZW4ga2V5Q29kZS5cbkF1ZGlvS2V5cy5wcm90b3R5cGUuX21hcCA9IGZ1bmN0aW9uKGtleUNvZGUpIHtcbiAgcmV0dXJuIHRoaXMuX2tleU1hcFt0aGlzLl9zdGF0ZS5yb3dzXVtrZXlDb2RlXSArIHRoaXMuX29mZnNldCgpO1xufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5fb2Zmc2V0ID0gZnVuY3Rpb24oKSB7XG4gIHJldHVybiB0aGlzLl9zdGF0ZS5yb290Tm90ZSAtIHRoaXMuX2tleU1hcFt0aGlzLl9zdGF0ZS5yb3dzXS5yb290ICsgKHRoaXMuX3N0YXRlLm9jdGF2ZSAqIDEyKTtcbn07XG5cbi8vIF9pc05vdGUgZGV0ZXJtaW5lcyB3aGV0aGVyIGEga2V5Q29kZSBpcyBhIG5vdGUgb3Igbm90LlxuQXVkaW9LZXlzLnByb3RvdHlwZS5faXNOb3RlID0gZnVuY3Rpb24oa2V5Q29kZSkge1xuICByZXR1cm4gISF0aGlzLl9rZXlNYXBbdGhpcy5fc3RhdGUucm93c11ba2V5Q29kZV07XG59O1xuXG4vLyBjb252ZXJ0IGEgbWlkaSBub3RlIHRvIGEgZnJlcXVlbmN5LiB3ZSBhc3N1bWUgaGVyZSB0aGF0IF9tYXAgaGFzXG4vLyBhbHJlYWR5IGJlZW4gY2FsbGVkICh0byBhY2NvdW50IGZvciBhIHBvdGVudGlhbCByb290Tm90ZSBvZmZzZXQpXG5BdWRpb0tleXMucHJvdG90eXBlLl90b0ZyZXF1ZW5jeSA9IGZ1bmN0aW9uKG5vdGUpIHtcbiAgcmV0dXJuICggTWF0aC5wb3coMiwgKCBub3RlLTY5ICkgLyAxMikgKSAqIDQ0MC4wO1xufTtcblxuLy8gdGhlIG9iamVjdCBrZXlzIGNvcnJlc3BvbmQgdG8gYHJvd3NgLCBzbyBgX2tleU1hcFtyb3dzXWAgc2hvdWxkXG4vLyByZXRyaWV2ZSB0aGF0IHBhcnRpY3VsYXIgbWFwcGluZy5cbkF1ZGlvS2V5cy5wcm90b3R5cGUuX2tleU1hcCA9IHtcbiAgMToge1xuICAgIHJvb3Q6IDYwLFxuICAgIC8vIHN0YXJ0aW5nIHdpdGggdGhlICdhJyBrZXlcbiAgICA2NTogIDYwLFxuICAgIDg3OiAgNjEsXG4gICAgODM6ICA2MixcbiAgICA2OTogIDYzLFxuICAgIDY4OiAgNjQsXG4gICAgNzA6ICA2NSxcbiAgICA4NDogIDY2LFxuICAgIDcxOiAgNjcsXG4gICAgODk6ICA2OCxcbiAgICA3MjogIDY5LFxuICAgIDg1OiAgNzAsXG4gICAgNzQ6ICA3MSxcbiAgICA3NTogIDcyLFxuICAgIDc5OiAgNzMsXG4gICAgNzY6ICA3NCxcbiAgICA4MDogIDc1LFxuICAgIDE4NjogNzYsXG4gICAgMjIyOiA3N1xuICB9LFxuICAyOiB7XG4gICAgcm9vdDogNjAsXG4gICAgLy8gYm90dG9tIHJvd1xuICAgIDkwOiAgNjAsXG4gICAgODM6ICA2MSxcbiAgICA4ODogIDYyLFxuICAgIDY4OiAgNjMsXG4gICAgNjc6ICA2NCxcbiAgICA4NjogIDY1LFxuICAgIDcxOiAgNjYsXG4gICAgNjY6ICA2NyxcbiAgICA3MjogIDY4LFxuICAgIDc4OiAgNjksXG4gICAgNzQ6ICA3MCxcbiAgICA3NzogIDcxLFxuICAgIDE4ODogNzIsXG4gICAgNzY6ICA3MyxcbiAgICAxOTA6IDc0LFxuICAgIDE4NjogNzUsXG4gICAgMTkxOiA3NixcbiAgICAvLyB0b3Agcm93XG4gICAgODE6ICA3MixcbiAgICA1MDogIDczLFxuICAgIDg3OiAgNzQsXG4gICAgNTE6ICA3NSxcbiAgICA2OTogIDc2LFxuICAgIDgyOiAgNzcsXG4gICAgNTM6ICA3OCxcbiAgICA4NDogIDc5LFxuICAgIDU0OiAgODAsXG4gICAgODk6ICA4MSxcbiAgICA1NTogIDgyLFxuICAgIDg1OiAgODMsXG4gICAgNzM6ICA4NCxcbiAgICA1NzogIDg1LFxuICAgIDc5OiAgODYsXG4gICAgNDg6ICA4NyxcbiAgICA4MDogIDg4LFxuICAgIDIxOTogODksXG4gICAgMTg3OiA5MCxcbiAgICAyMjE6IDkxXG4gIH1cbn07XG4iLCIvLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4vLyBLRVkgQlVGRkVSXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG5cbi8vIFRoZSBwcm9jZXNzIGlzOlxuXG4vLyBrZXkgcHJlc3Ncbi8vICAgYWRkIHRvIHNlbGYuX3N0YXRlLmtleXNcbi8vICAgKGFuIGFjY3VyYXRlIHJlcHJlc2VudGF0aW9uIG9mIGtleXMgY3VycmVudGx5IHByZXNzZWQpXG4vLyByZXNvbHZlIHNlbGYuYnVmZmVyXG4vLyAgIGJhc2VkIG9uIHBvbHlwaG9ueSBhbmQgcHJpb3JpdHksIGRldGVybWluZSB0aGUgbm90ZXNcbi8vICAgdGhhdCBnZXQgdHJpZ2dlcmVkIGZvciB0aGUgdXNlclxuXG5BdWRpb0tleXMucHJvdG90eXBlLl9hZGRLZXkgPSBmdW5jdGlvbihlKSB7XG4gIHZhciBzZWxmID0gdGhpcztcbiAgLy8gaWYgdGhlIGtleUNvZGUgaXMgb25lIHRoYXQgY2FuIGJlIG1hcHBlZCBhbmQgaXNuJ3RcbiAgLy8gYWxyZWFkeSBwcmVzc2VkLCBhZGQgaXQgdG8gdGhlIGtleSBvYmplY3QuXG4gIGlmKHNlbGYuX2lzTm90ZShlLmtleUNvZGUpICYmICFzZWxmLl9pc1ByZXNzZWQoZS5rZXlDb2RlKSkge1xuICAgIHZhciBuZXdLZXkgPSBzZWxmLl9tYWtlTm90ZShlLmtleUNvZGUpO1xuICAgIC8vIGFkZCB0aGUgbmV3S2V5IHRvIHRoZSBsaXN0IG9mIGtleXNcbiAgICBzZWxmLl9zdGF0ZS5rZXlzID0gKHNlbGYuX3N0YXRlLmtleXMgfHwgW10pLmNvbmNhdChuZXdLZXkpO1xuICAgIC8vIHJlZXZhbHVhdGUgdGhlIGFjdGl2ZSBub3RlcyBiYXNlZCBvbiBvdXIgcHJpb3JpdHkgcnVsZXMuXG4gICAgLy8gZ2l2ZSBpdCB0aGUgbmV3IG5vdGUgdG8gdXNlIGlmIHRoZXJlIGlzIGFuIGV2ZW50IHRvIHRyaWdnZXIuXG4gICAgc2VsZi5fdXBkYXRlKCk7XG4gIH0gZWxzZSBpZihzZWxmLl9pc1NwZWNpYWxLZXkoZS5rZXlDb2RlKSkge1xuICAgIHNlbGYuX3NwZWNpYWxLZXkoZS5rZXlDb2RlKTtcbiAgfVxufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5fcmVtb3ZlS2V5ID0gZnVuY3Rpb24oZSkge1xuICB2YXIgc2VsZiA9IHRoaXM7XG4gIC8vIGlmIHRoZSBrZXlDb2RlIGlzIGFjdGl2ZSwgcmVtb3ZlIGl0IGZyb20gdGhlIGtleSBvYmplY3QuXG4gIGlmKHNlbGYuX2lzUHJlc3NlZChlLmtleUNvZGUpKSB7XG4gICAgdmFyIGtleVRvUmVtb3ZlO1xuICAgIGZvcih2YXIgaSA9IDA7IGkgPCBzZWxmLl9zdGF0ZS5rZXlzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBpZihzZWxmLl9zdGF0ZS5rZXlzW2ldLmtleUNvZGUgPT09IGUua2V5Q29kZSkge1xuICAgICAgICBrZXlUb1JlbW92ZSA9IHNlbGYuX3N0YXRlLmtleXNbaV07XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIHJlbW92ZSB0aGUga2V5IGZyb20gX2tleXNcbiAgICBzZWxmLl9zdGF0ZS5rZXlzLnNwbGljZShzZWxmLl9zdGF0ZS5rZXlzLmluZGV4T2Yoa2V5VG9SZW1vdmUpLCAxKTtcbiAgICBzZWxmLl91cGRhdGUoKTtcbiAgfVxufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5faXNQcmVzc2VkID0gZnVuY3Rpb24oa2V5Q29kZSkge1xuICB2YXIgc2VsZiA9IHRoaXM7XG5cbiAgaWYoIXNlbGYuX3N0YXRlLmtleXMgfHwgIXNlbGYuX3N0YXRlLmtleXMubGVuZ3RoKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgZm9yKHZhciBpID0gMDsgaSA8IHNlbGYuX3N0YXRlLmtleXMubGVuZ3RoOyBpKyspIHtcbiAgICBpZihzZWxmLl9zdGF0ZS5rZXlzW2ldLmtleUNvZGUgPT09IGtleUNvZGUpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cbiAgfVxuICByZXR1cm4gZmFsc2U7XG59O1xuXG4vLyB0dXJuIGEga2V5IG9iamVjdCBpbnRvIGEgbm90ZSBvYmplY3QgZm9yIHRoZSBldmVudCBsaXN0ZW5lcnMuXG5BdWRpb0tleXMucHJvdG90eXBlLl9tYWtlTm90ZSA9IGZ1bmN0aW9uKGtleUNvZGUpIHtcbiAgdmFyIHNlbGYgPSB0aGlzO1xuICByZXR1cm4ge1xuICAgIGtleUNvZGU6IGtleUNvZGUsXG4gICAgbm90ZTogc2VsZi5fbWFwKGtleUNvZGUpLFxuICAgIGZyZXF1ZW5jeTogc2VsZi5fdG9GcmVxdWVuY3koIHNlbGYuX21hcChrZXlDb2RlKSApLFxuICAgIHZlbG9jaXR5OiBzZWxmLl9zdGF0ZS52ZWxvY2l0eVxuICB9O1xufTtcblxuLy8gY2xlYXIgYW55IGFjdGl2ZSBub3Rlc1xuQXVkaW9LZXlzLnByb3RvdHlwZS5jbGVhciA9IGZ1bmN0aW9uKCkge1xuICB2YXIgc2VsZiA9IHRoaXM7XG4gIC8vIHRyaWdnZXIgbm90ZSBvZmYgZm9yIHRoZSBub3RlcyBpbiB0aGUgYnVmZmVyIGJlZm9yZVxuICAvLyByZW1vdmluZyB0aGVtLlxuICBzZWxmLl9zdGF0ZS5idWZmZXIuZm9yRWFjaCggZnVuY3Rpb24oa2V5KSB7XG4gICAgc2VsZi5fdHJpZ2dlcigndXAnLCBrZXkpO1xuICB9KTtcbiAgc2VsZi5fc3RhdGUua2V5cyA9IFtdO1xuICBzZWxmLl9zdGF0ZS5idWZmZXIgPSBbXTtcbn07XG5cbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbi8vIE5PVEUgQlVGRkVSXG4vLyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG5cbi8vIGV2ZXJ5IHRpbWUgYSBjaGFuZ2UgaXMgbWFkZSB0byBfa2V5cyBkdWUgdG8gYSBrZXkgb24gb3Iga2V5IG9mZlxuLy8gd2UgbmVlZCB0byBjYWxsIGBfdXBkYXRlYC4gSXQgY29tcGFyZXMgdGhlIGBfa2V5c2AgYXJyYXkgdG8gdGhlXG4vLyBgYnVmZmVyYCBhcnJheSwgd2hpY2ggaXMgdGhlIGFycmF5IG9mIG5vdGVzIHRoYXQgYXJlIHJlYWxseVxuLy8gYmVpbmcgcGxheWVkLCBtYWtlcyB0aGUgbmVjZXNzYXJ5IGNoYW5nZXMgdG8gYGJ1ZmZlcmAgYW5kXG4vLyB0cmlnZ2VycyBhbnkgZXZlbnRzIHRoYXQgbmVlZCB0cmlnZ2VyaW5nLlxuXG5BdWRpb0tleXMucHJvdG90eXBlLl91cGRhdGUgPSBmdW5jdGlvbigpIHtcbiAgdmFyIHNlbGYgPSB0aGlzO1xuXG4gIC8vIGEga2V5IGhhcyBiZWVuIGFkZGVkIHRvIHNlbGYuX3N0YXRlLmtleXMuXG4gIC8vIHN0YXNoIHRoZSBvbGQgYnVmZmVyXG4gIHZhciBvbGRCdWZmZXIgPSBzZWxmLl9zdGF0ZS5idWZmZXI7XG4gIC8vIHNldCB0aGUgbmV3IHByaW9yaXR5IGluIHNlbGYuc3RhdGUuX2tleXNcbiAgc2VsZi5fcHJpb3JpdGl6ZSgpO1xuICAvLyBjb21wYXJlIHRoZSBidWZmZXJzIGFuZCB0cmlnZ2VyIGV2ZW50cyBiYXNlZCBvblxuICAvLyB0aGUgZGlmZmVyZW5jZXMuXG4gIHNlbGYuX2RpZmYob2xkQnVmZmVyKTtcbn07XG5cbkF1ZGlvS2V5cy5wcm90b3R5cGUuX2RpZmYgPSBmdW5jdGlvbihvbGRCdWZmZXIpIHtcbiAgdmFyIHNlbGYgPSB0aGlzO1xuXG4gIC8vIGlmIGl0J3Mgbm90IGluIHRoZSBPTEQgYnVmZmVyLCBpdCdzIGEgbm90ZSBPTi5cbiAgLy8gaWYgaXQncyBub3QgaW4gdGhlIE5FVyBidWZmZXIsIGl0J3MgYSBub3RlIE9GRi5cblxuICB2YXIgb2xkTm90ZXMgPSBvbGRCdWZmZXIubWFwKCBmdW5jdGlvbihrZXkpIHtcbiAgICByZXR1cm4ga2V5LmtleUNvZGU7XG4gIH0pO1xuXG4gIHZhciBuZXdOb3RlcyA9IHNlbGYuX3N0YXRlLmJ1ZmZlci5tYXAoIGZ1bmN0aW9uKGtleSkge1xuICAgIHJldHVybiBrZXkua2V5Q29kZTtcbiAgfSk7XG5cbiAgLy8gY2hlY2sgZm9yIG9sZCAocmVtb3ZlZCkgbm90ZXNcbiAgdmFyIG5vdGVzVG9SZW1vdmUgPSBbXTtcbiAgb2xkTm90ZXMuZm9yRWFjaCggZnVuY3Rpb24oa2V5KSB7XG4gICAgaWYobmV3Tm90ZXMuaW5kZXhPZihrZXkpID09PSAtMSkge1xuICAgICAgbm90ZXNUb1JlbW92ZS5wdXNoKGtleSk7XG4gICAgfVxuICB9KTtcblxuICAvLyBjaGVjayBmb3IgbmV3IG5vdGVzXG4gIHZhciBub3Rlc1RvQWRkID0gW107XG4gIG5ld05vdGVzLmZvckVhY2goIGZ1bmN0aW9uKGtleSkge1xuICAgIGlmKG9sZE5vdGVzLmluZGV4T2Yoa2V5KSA9PT0gLTEpIHtcbiAgICAgIG5vdGVzVG9BZGQucHVzaChrZXkpO1xuICAgIH1cbiAgfSk7XG5cbiAgbm90ZXNUb0FkZC5mb3JFYWNoKCBmdW5jdGlvbihrZXkpIHtcbiAgICBmb3IodmFyIGkgPSAwOyBpIDwgc2VsZi5fc3RhdGUuYnVmZmVyLmxlbmd0aDsgaSsrKSB7XG4gICAgICBpZihzZWxmLl9zdGF0ZS5idWZmZXJbaV0ua2V5Q29kZSA9PT0ga2V5KSB7XG4gICAgICAgIHNlbGYuX3RyaWdnZXIoJ2Rvd24nLCBzZWxmLl9zdGF0ZS5idWZmZXJbaV0pO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICB9XG4gIH0pO1xuXG4gIG5vdGVzVG9SZW1vdmUuZm9yRWFjaCggZnVuY3Rpb24oa2V5KSB7XG4gICAgLy8gdGhlc2UgbmVlZCB0byBmaXJlIHRoZSBlbnRpcmUgb2JqZWN0XG4gICAgZm9yKHZhciBpID0gMDsgaSA8IG9sZEJ1ZmZlci5sZW5ndGg7IGkrKykge1xuICAgICAgaWYob2xkQnVmZmVyW2ldLmtleUNvZGUgPT09IGtleSkge1xuICAgICAgICBzZWxmLl90cmlnZ2VyKCd1cCcsIG9sZEJ1ZmZlcltpXSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgIH1cbiAgfSk7XG59O1xuIiwiQXVkaW9LZXlzLnByb3RvdHlwZS5fcHJpb3JpdGl6ZSA9IGZ1bmN0aW9uKCkge1xuICB2YXIgc2VsZiA9IHRoaXM7XG5cbiAgLy8gaWYgYWxsIHRoZSBrZXlzIGhhdmUgYmVlbiB0dXJuZWQgb2ZmLCBubyBuZWVkXG4gIC8vIHRvIGRvIGFueXRoaW5nIGhlcmUuXG4gIGlmKCFzZWxmLl9zdGF0ZS5rZXlzLmxlbmd0aCkge1xuICAgIHNlbGYuX3N0YXRlLmJ1ZmZlciA9IFtdO1xuICAgIHJldHVybjtcbiAgfVxuXG5cbiAgaWYoc2VsZi5fc3RhdGUucG9seXBob255ID49IHNlbGYuX3N0YXRlLmtleXMubGVuZ3RoKSB7XG4gICAgLy8gZXZlcnkgbm90ZSBpcyBhY3RpdmVcbiAgICBzZWxmLl9zdGF0ZS5rZXlzID0gc2VsZi5fc3RhdGUua2V5cy5tYXAoIGZ1bmN0aW9uKGtleSkge1xuICAgICAga2V5LmlzQWN0aXZlID0gdHJ1ZTtcbiAgICAgIHJldHVybiBrZXk7XG4gICAgfSk7XG4gIH0gZWxzZSB7XG4gICAgLy8gc2V0IGFsbCBrZXlzIHRvIGluYWN0aXZlLlxuICAgIHNlbGYuX3N0YXRlLmtleXMgPSBzZWxmLl9zdGF0ZS5rZXlzLm1hcCggZnVuY3Rpb24oa2V5KSB7XG4gICAgICBrZXkuaXNBY3RpdmUgPSBmYWxzZTtcbiAgICAgIHJldHVybiBrZXk7XG4gICAgfSk7XG5cbiAgICBzZWxmWydfJyArIHNlbGYuX3N0YXRlLnByaW9yaXR5XSgpO1xuICB9XG5cbiAgLy8gbm93IHRha2UgdGhlIGlzQWN0aXZlIGtleXMgYW5kIHNldCB0aGUgbmV3IGJ1ZmZlci5cbiAgc2VsZi5fc3RhdGUuYnVmZmVyID0gW107XG5cbiAgc2VsZi5fc3RhdGUua2V5cy5mb3JFYWNoKCBmdW5jdGlvbihrZXkpIHtcbiAgICBpZihrZXkuaXNBY3RpdmUpIHtcbiAgICAgIHNlbGYuX3N0YXRlLmJ1ZmZlci5wdXNoKGtleSk7XG4gICAgfVxuICB9KTtcblxuICAvLyBkb25lLlxufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5fbGFzdCA9IGZ1bmN0aW9uKCkge1xuICB2YXIgc2VsZiA9IHRoaXM7XG4gIC8vIHNldCB0aGUgbGFzdCBidW5jaCB0byBhY3RpdmUgYmFzZWQgb24gdGhlIHBvbHlwaG9ueS5cbiAgZm9yKHZhciBpID0gc2VsZi5fc3RhdGUua2V5cy5sZW5ndGggLSBzZWxmLl9zdGF0ZS5wb2x5cGhvbnk7IGkgPCBzZWxmLl9zdGF0ZS5rZXlzLmxlbmd0aDsgaSsrKSB7XG4gICAgc2VsZi5fc3RhdGUua2V5c1tpXS5pc0FjdGl2ZSA9IHRydWU7XG4gIH1cbn07XG5cbkF1ZGlvS2V5cy5wcm90b3R5cGUuX2ZpcnN0ID0gZnVuY3Rpb24oKSB7XG4gIHZhciBzZWxmID0gdGhpcztcbiAgLy8gc2V0IHRoZSBsYXN0IGJ1bmNoIHRvIGFjdGl2ZSBiYXNlZCBvbiB0aGUgcG9seXBob255LlxuICBmb3IodmFyIGkgPSAwOyBpIDwgc2VsZi5fc3RhdGUucG9seXBob255OyBpKyspIHtcbiAgICBzZWxmLl9zdGF0ZS5rZXlzW2ldLmlzQWN0aXZlID0gdHJ1ZTtcbiAgfVxufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5faGlnaGVzdCA9IGZ1bmN0aW9uKCkge1xuICB2YXIgc2VsZiA9IHRoaXM7XG4gIC8vIGdldCB0aGUgaGlnaGVzdCBub3RlcyBhbmQgc2V0IHRoZW0gdG8gYWN0aXZlXG4gIHZhciBub3RlcyA9IHNlbGYuX3N0YXRlLmtleXMubWFwKCBmdW5jdGlvbihrZXkpIHtcbiAgICByZXR1cm4ga2V5Lm5vdGU7XG4gIH0pO1xuXG4gIG5vdGVzLnNvcnQoIGZ1bmN0aW9uKGIsYSkge1xuICAgIGlmKGEgPT09IGIpIHtcbiAgICAgIHJldHVybiAwO1xuICAgIH1cbiAgICByZXR1cm4gYSA8IGIgPyAtMSA6IDE7XG4gIH0pO1xuXG4gIG5vdGVzLnNwbGljZShzZWxmLl9zdGF0ZS5wb2x5cGhvbnksIE51bWJlci5NQVhfVkFMVUUpO1xuXG4gIHNlbGYuX3N0YXRlLmtleXMuZm9yRWFjaCggZnVuY3Rpb24oa2V5KSB7XG4gICAgaWYobm90ZXMuaW5kZXhPZihrZXkubm90ZSkgIT09IC0xKSB7XG4gICAgICBrZXkuaXNBY3RpdmUgPSB0cnVlO1xuICAgIH1cbiAgfSk7XG59O1xuXG5BdWRpb0tleXMucHJvdG90eXBlLl9sb3dlc3QgPSBmdW5jdGlvbigpIHtcbiAgdmFyIHNlbGYgPSB0aGlzO1xuICAvLyBnZXQgdGhlIGxvd2VzdCBub3RlcyBhbmQgc2V0IHRoZW0gdG8gYWN0aXZlXG4gIHZhciBub3RlcyA9IHNlbGYuX3N0YXRlLmtleXMubWFwKCBmdW5jdGlvbihrZXkpIHtcbiAgICByZXR1cm4ga2V5Lm5vdGU7XG4gIH0pO1xuXG4gIG5vdGVzLnNvcnQoIGZ1bmN0aW9uKGEsYikge1xuICAgIGlmKGEgPT09IGIpIHtcbiAgICAgIHJldHVybiAwO1xuICAgIH1cbiAgICByZXR1cm4gYSA8IGIgPyAtMSA6IDE7XG4gIH0pO1xuXG4gIG5vdGVzLnNwbGljZShzZWxmLl9zdGF0ZS5wb2x5cGhvbnksIE51bWJlci5NQVhfVkFMVUUpO1xuXG4gIHNlbGYuX3N0YXRlLmtleXMuZm9yRWFjaCggZnVuY3Rpb24oa2V5KSB7XG4gICAgaWYobm90ZXMuaW5kZXhPZihrZXkubm90ZSkgIT09IC0xKSB7XG4gICAgICBrZXkuaXNBY3RpdmUgPSB0cnVlO1xuICAgIH1cbiAgfSk7XG59O1xuIiwiLy8gVGhpcyBmaWxlIG1hcHMgc3BlY2lhbCBrZXlzIHRvIHRoZSBzdGF0ZeKAlCBvY3RhdmUgc2hpZnRpbmcgYW5kXG4vLyB2ZWxvY2l0eSBzZWxlY3Rpb24sIGJvdGggYXZhaWxhYmxlIHdoZW4gYHJvd3NgID0gMS5cblxuQXVkaW9LZXlzLnByb3RvdHlwZS5faXNTcGVjaWFsS2V5ID0gZnVuY3Rpb24oa2V5Q29kZSkge1xuICByZXR1cm4gKHRoaXMuX3N0YXRlLnJvd3MgPT09IDEgJiYgdGhpcy5fc3BlY2lhbEtleU1hcFtrZXlDb2RlXSk7XG59O1xuXG5BdWRpb0tleXMucHJvdG90eXBlLl9zcGVjaWFsS2V5ID0gZnVuY3Rpb24oa2V5Q29kZSkge1xuICB2YXIgc2VsZiA9IHRoaXM7XG4gIGlmKHNlbGYuX3NwZWNpYWxLZXlNYXBba2V5Q29kZV0udHlwZSA9PT0gJ29jdGF2ZScgJiYgc2VsZi5fc3RhdGUub2N0YXZlQ29udHJvbHMpIHtcbiAgICAvLyBzaGlmdCB0aGUgc3RhdGUgb2YgdGhlIGBvY3RhdmVgXG4gICAgc2VsZi5fc3RhdGUub2N0YXZlICs9IHNlbGYuX3NwZWNpYWxLZXlNYXBba2V5Q29kZV0udmFsdWU7XG4gIH0gZWxzZSBpZihzZWxmLl9zcGVjaWFsS2V5TWFwW2tleUNvZGVdLnR5cGUgPT09ICd2ZWxvY2l0eScgJiYgc2VsZi5fc3RhdGUudmVsb2NpdHlDb250cm9scykge1xuICAgIC8vIHNldCB0aGUgYHZlbG9jaXR5YCB0byBhIG5ldyB2YWx1ZVxuICAgIHNlbGYuX3N0YXRlLnZlbG9jaXR5ID0gc2VsZi5fc3BlY2lhbEtleU1hcFtrZXlDb2RlXS52YWx1ZTtcbiAgfVxufTtcblxuQXVkaW9LZXlzLnByb3RvdHlwZS5fc3BlY2lhbEtleU1hcCA9IHtcbiAgLy8gb2N0YXZlc1xuICA5MDoge1xuICAgIHR5cGU6ICdvY3RhdmUnLFxuICAgIHZhbHVlOiAtMVxuICB9LFxuICA4ODoge1xuICAgIHR5cGU6ICdvY3RhdmUnLFxuICAgIHZhbHVlOiAxXG4gIH0sXG4gIC8vIHZlbG9jaXR5XG4gIDQ5OiB7XG4gICAgdHlwZTogJ3ZlbG9jaXR5JyxcbiAgICB2YWx1ZTogMVxuICB9LFxuICA1MDoge1xuICAgIHR5cGU6ICd2ZWxvY2l0eScsXG4gICAgdmFsdWU6IDE0XG4gIH0sXG4gIDUxOiB7XG4gICAgdHlwZTogJ3ZlbG9jaXR5JyxcbiAgICB2YWx1ZTogMjhcbiAgfSxcbiAgNTI6IHtcbiAgICB0eXBlOiAndmVsb2NpdHknLFxuICAgIHZhbHVlOiA0MlxuICB9LFxuICA1Mzoge1xuICAgIHR5cGU6ICd2ZWxvY2l0eScsXG4gICAgdmFsdWU6IDU2XG4gIH0sXG4gIDU0OiB7XG4gICAgdHlwZTogJ3ZlbG9jaXR5JyxcbiAgICB2YWx1ZTogNzBcbiAgfSxcbiAgNTU6IHtcbiAgICB0eXBlOiAndmVsb2NpdHknLFxuICAgIHZhbHVlOiA4NFxuICB9LFxuICA1Njoge1xuICAgIHR5cGU6ICd2ZWxvY2l0eScsXG4gICAgdmFsdWU6IDk4XG4gIH0sXG4gIDU3OiB7XG4gICAgdHlwZTogJ3ZlbG9jaXR5JyxcbiAgICB2YWx1ZTogMTEyXG4gIH0sXG4gIDQ4OiB7XG4gICAgdHlwZTogJ3ZlbG9jaXR5JyxcbiAgICB2YWx1ZTogMTI3XG4gIH0sXG59O1xuIl0sInNvdXJjZVJvb3QiOiIvc291cmNlLyJ9