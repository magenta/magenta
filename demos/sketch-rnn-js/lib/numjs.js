/**The MIT License (MIT)
 *
 * Copyright (c) 2016 Nicolas Panel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/**
 * Author: Nicolas Panel
 *
 * @fileoverview See https://github.com/nicolaspanel/numjs/
 */
(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.nj = f()}})(function(){var define,module,exports;return (function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
'use strict'

exports.toByteArray = toByteArray
exports.fromByteArray = fromByteArray

var lookup = []
var revLookup = []
var Arr = typeof Uint8Array !== 'undefined' ? Uint8Array : Array

function init () {
  var code = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
  for (var i = 0, len = code.length; i < len; ++i) {
    lookup[i] = code[i]
    revLookup[code.charCodeAt(i)] = i
  }

  revLookup['-'.charCodeAt(0)] = 62
  revLookup['_'.charCodeAt(0)] = 63
}

init()

function toByteArray (b64) {
  var i, j, l, tmp, placeHolders, arr
  var len = b64.length

  if (len % 4 > 0) {
    throw new Error('Invalid string. Length must be a multiple of 4')
  }

  // the number of equal signs (place holders)
  // if there are two placeholders, than the two characters before it
  // represent one byte
  // if there is only one, then the three characters before it represent 2 bytes
  // this is just a cheap hack to not do indexOf twice
  placeHolders = b64[len - 2] === '=' ? 2 : b64[len - 1] === '=' ? 1 : 0

  // base64 is 4/3 + up to two characters of the original data
  arr = new Arr(len * 3 / 4 - placeHolders)

  // if there are placeholders, only get up to the last complete 4 chars
  l = placeHolders > 0 ? len - 4 : len

  var L = 0

  for (i = 0, j = 0; i < l; i += 4, j += 3) {
    tmp = (revLookup[b64.charCodeAt(i)] << 18) | (revLookup[b64.charCodeAt(i + 1)] << 12) | (revLookup[b64.charCodeAt(i + 2)] << 6) | revLookup[b64.charCodeAt(i + 3)]
    arr[L++] = (tmp >> 16) & 0xFF
    arr[L++] = (tmp >> 8) & 0xFF
    arr[L++] = tmp & 0xFF
  }

  if (placeHolders === 2) {
    tmp = (revLookup[b64.charCodeAt(i)] << 2) | (revLookup[b64.charCodeAt(i + 1)] >> 4)
    arr[L++] = tmp & 0xFF
  } else if (placeHolders === 1) {
    tmp = (revLookup[b64.charCodeAt(i)] << 10) | (revLookup[b64.charCodeAt(i + 1)] << 4) | (revLookup[b64.charCodeAt(i + 2)] >> 2)
    arr[L++] = (tmp >> 8) & 0xFF
    arr[L++] = tmp & 0xFF
  }

  return arr
}

function tripletToBase64 (num) {
  return lookup[num >> 18 & 0x3F] + lookup[num >> 12 & 0x3F] + lookup[num >> 6 & 0x3F] + lookup[num & 0x3F]
}

function encodeChunk (uint8, start, end) {
  var tmp
  var output = []
  for (var i = start; i < end; i += 3) {
    tmp = (uint8[i] << 16) + (uint8[i + 1] << 8) + (uint8[i + 2])
    output.push(tripletToBase64(tmp))
  }
  return output.join('')
}

function fromByteArray (uint8) {
  var tmp
  var len = uint8.length
  var extraBytes = len % 3 // if we have 1 byte left, pad 2 bytes
  var output = ''
  var parts = []
  var maxChunkLength = 16383 // must be multiple of 3

  // go through the array every three bytes, we'll deal with trailing stuff later
  for (var i = 0, len2 = len - extraBytes; i < len2; i += maxChunkLength) {
    parts.push(encodeChunk(uint8, i, (i + maxChunkLength) > len2 ? len2 : (i + maxChunkLength)))
  }

  // pad the end with zeros, but make sure to not forget the extra bytes
  if (extraBytes === 1) {
    tmp = uint8[len - 1]
    output += lookup[tmp >> 2]
    output += lookup[(tmp << 4) & 0x3F]
    output += '=='
  } else if (extraBytes === 2) {
    tmp = (uint8[len - 2] << 8) + (uint8[len - 1])
    output += lookup[tmp >> 10]
    output += lookup[(tmp >> 4) & 0x3F]
    output += lookup[(tmp << 2) & 0x3F]
    output += '='
  }

  parts.push(output)

  return parts.join('')
}

},{}],2:[function(require,module,exports){
/**
 * Bit twiddling hacks for JavaScript.
 *
 * Author: Mikola Lysenko
 *
 * Ported from Stanford bit twiddling hack library:
 *    http://graphics.stanford.edu/~seander/bithacks.html
 */

"use strict"; "use restrict";

//Number of bits in an integer
var INT_BITS = 32;

//Constants
exports.INT_BITS  = INT_BITS;
exports.INT_MAX   =  0x7fffffff;
exports.INT_MIN   = -1<<(INT_BITS-1);

//Returns -1, 0, +1 depending on sign of x
exports.sign = function(v) {
  return (v > 0) - (v < 0);
}

//Computes absolute value of integer
exports.abs = function(v) {
  var mask = v >> (INT_BITS-1);
  return (v ^ mask) - mask;
}

//Computes minimum of integers x and y
exports.min = function(x, y) {
  return y ^ ((x ^ y) & -(x < y));
}

//Computes maximum of integers x and y
exports.max = function(x, y) {
  return x ^ ((x ^ y) & -(x < y));
}

//Checks if a number is a power of two
exports.isPow2 = function(v) {
  return !(v & (v-1)) && (!!v);
}

//Computes log base 2 of v
exports.log2 = function(v) {
  var r, shift;
  r =     (v > 0xFFFF) << 4; v >>>= r;
  shift = (v > 0xFF  ) << 3; v >>>= shift; r |= shift;
  shift = (v > 0xF   ) << 2; v >>>= shift; r |= shift;
  shift = (v > 0x3   ) << 1; v >>>= shift; r |= shift;
  return r | (v >> 1);
}

//Computes log base 10 of v
exports.log10 = function(v) {
  return  (v >= 1000000000) ? 9 : (v >= 100000000) ? 8 : (v >= 10000000) ? 7 :
          (v >= 1000000) ? 6 : (v >= 100000) ? 5 : (v >= 10000) ? 4 :
          (v >= 1000) ? 3 : (v >= 100) ? 2 : (v >= 10) ? 1 : 0;
}

//Counts number of bits
exports.popCount = function(v) {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return ((v + (v >>> 4) & 0xF0F0F0F) * 0x1010101) >>> 24;
}

//Counts number of trailing zeros
function countTrailingZeros(v) {
  var c = 32;
  v &= -v;
  if (v) c--;
  if (v & 0x0000FFFF) c -= 16;
  if (v & 0x00FF00FF) c -= 8;
  if (v & 0x0F0F0F0F) c -= 4;
  if (v & 0x33333333) c -= 2;
  if (v & 0x55555555) c -= 1;
  return c;
}
exports.countTrailingZeros = countTrailingZeros;

//Rounds to next power of 2
exports.nextPow2 = function(v) {
  v += v === 0;
  --v;
  v |= v >>> 1;
  v |= v >>> 2;
  v |= v >>> 4;
  v |= v >>> 8;
  v |= v >>> 16;
  return v + 1;
}

//Rounds down to previous power of 2
exports.prevPow2 = function(v) {
  v |= v >>> 1;
  v |= v >>> 2;
  v |= v >>> 4;
  v |= v >>> 8;
  v |= v >>> 16;
  return v - (v>>>1);
}

//Computes parity of word
exports.parity = function(v) {
  v ^= v >>> 16;
  v ^= v >>> 8;
  v ^= v >>> 4;
  v &= 0xf;
  return (0x6996 >>> v) & 1;
}

var REVERSE_TABLE = new Array(256);

(function(tab) {
  for(var i=0; i<256; ++i) {
    var v = i, r = i, s = 7;
    for (v >>>= 1; v; v >>>= 1) {
      r <<= 1;
      r |= v & 1;
      --s;
    }
    tab[i] = (r << s) & 0xff;
  }
})(REVERSE_TABLE);

//Reverse bits in a 32 bit word
exports.reverse = function(v) {
  return  (REVERSE_TABLE[ v         & 0xff] << 24) |
          (REVERSE_TABLE[(v >>> 8)  & 0xff] << 16) |
          (REVERSE_TABLE[(v >>> 16) & 0xff] << 8)  |
           REVERSE_TABLE[(v >>> 24) & 0xff];
}

//Interleave bits of 2 coordinates with 16 bits.  Useful for fast quadtree codes
exports.interleave2 = function(x, y) {
  x &= 0xFFFF;
  x = (x | (x << 8)) & 0x00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F;
  x = (x | (x << 2)) & 0x33333333;
  x = (x | (x << 1)) & 0x55555555;

  y &= 0xFFFF;
  y = (y | (y << 8)) & 0x00FF00FF;
  y = (y | (y << 4)) & 0x0F0F0F0F;
  y = (y | (y << 2)) & 0x33333333;
  y = (y | (y << 1)) & 0x55555555;

  return x | (y << 1);
}

//Extracts the nth interleaved component
exports.deinterleave2 = function(v, n) {
  v = (v >>> n) & 0x55555555;
  v = (v | (v >>> 1))  & 0x33333333;
  v = (v | (v >>> 2))  & 0x0F0F0F0F;
  v = (v | (v >>> 4))  & 0x00FF00FF;
  v = (v | (v >>> 16)) & 0x000FFFF;
  return (v << 16) >> 16;
}


//Interleave bits of 3 coordinates, each with 10 bits.  Useful for fast octree codes
exports.interleave3 = function(x, y, z) {
  x &= 0x3FF;
  x  = (x | (x<<16)) & 4278190335;
  x  = (x | (x<<8))  & 251719695;
  x  = (x | (x<<4))  & 3272356035;
  x  = (x | (x<<2))  & 1227133513;

  y &= 0x3FF;
  y  = (y | (y<<16)) & 4278190335;
  y  = (y | (y<<8))  & 251719695;
  y  = (y | (y<<4))  & 3272356035;
  y  = (y | (y<<2))  & 1227133513;
  x |= (y << 1);
  
  z &= 0x3FF;
  z  = (z | (z<<16)) & 4278190335;
  z  = (z | (z<<8))  & 251719695;
  z  = (z | (z<<4))  & 3272356035;
  z  = (z | (z<<2))  & 1227133513;
  
  return x | (z << 2);
}

//Extracts nth interleaved component of a 3-tuple
exports.deinterleave3 = function(v, n) {
  v = (v >>> n)       & 1227133513;
  v = (v | (v>>>2))   & 3272356035;
  v = (v | (v>>>4))   & 251719695;
  v = (v | (v>>>8))   & 4278190335;
  v = (v | (v>>>16))  & 0x3FF;
  return (v<<22)>>22;
}

//Computes next combination in colexicographic order (this is mistakenly called nextPermutation on the bit twiddling hacks page)
exports.nextCombination = function(v) {
  var t = v | (v - 1);
  return (t + 1) | (((~t & -~t) - 1) >>> (countTrailingZeros(v) + 1));
}


},{}],3:[function(require,module,exports){
(function (global){
/*!
 * The buffer module from node.js, for the browser.
 *
 * @author   Feross Aboukhadijeh <feross@feross.org> <http://feross.org>
 * @license  MIT
 */
/* eslint-disable no-proto */

'use strict'

var base64 = require('base64-js')
var ieee754 = require('ieee754')
var isArray = require('isarray')

exports.Buffer = Buffer
exports.SlowBuffer = SlowBuffer
exports.INSPECT_MAX_BYTES = 50

/**
 * If `Buffer.TYPED_ARRAY_SUPPORT`:
 *   === true    Use Uint8Array implementation (fastest)
 *   === false   Use Object implementation (most compatible, even IE6)
 *
 * Browsers that support typed arrays are IE 10+, Firefox 4+, Chrome 7+, Safari 5.1+,
 * Opera 11.6+, iOS 4.2+.
 *
 * Due to various browser bugs, sometimes the Object implementation will be used even
 * when the browser supports typed arrays.
 *
 * Note:
 *
 *   - Firefox 4-29 lacks support for adding new properties to `Uint8Array` instances,
 *     See: https://bugzilla.mozilla.org/show_bug.cgi?id=695438.
 *
 *   - Chrome 9-10 is missing the `TypedArray.prototype.subarray` function.
 *
 *   - IE10 has a broken `TypedArray.prototype.subarray` function which returns arrays of
 *     incorrect length in some situations.

 * We detect these buggy browsers and set `Buffer.TYPED_ARRAY_SUPPORT` to `false` so they
 * get the Object implementation, which is slower but behaves correctly.
 */
Buffer.TYPED_ARRAY_SUPPORT = global.TYPED_ARRAY_SUPPORT !== undefined
  ? global.TYPED_ARRAY_SUPPORT
  : typedArraySupport()

/*
 * Export kMaxLength after typed array support is determined.
 */
exports.kMaxLength = kMaxLength()

function typedArraySupport () {
  try {
    var arr = new Uint8Array(1)
    arr.__proto__ = {__proto__: Uint8Array.prototype, foo: function () { return 42 }}
    return arr.foo() === 42 && // typed array instances can be augmented
        typeof arr.subarray === 'function' && // chrome 9-10 lack `subarray`
        arr.subarray(1, 1).byteLength === 0 // ie10 has broken `subarray`
  } catch (e) {
    return false
  }
}

function kMaxLength () {
  return Buffer.TYPED_ARRAY_SUPPORT
    ? 0x7fffffff
    : 0x3fffffff
}

function createBuffer (that, length) {
  if (kMaxLength() < length) {
    throw new RangeError('Invalid typed array length')
  }
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    // Return an augmented `Uint8Array` instance, for best performance
    that = new Uint8Array(length)
    that.__proto__ = Buffer.prototype
  } else {
    // Fallback: Return an object instance of the Buffer class
    if (that === null) {
      that = new Buffer(length)
    }
    that.length = length
  }

  return that
}

/**
 * The Buffer constructor returns instances of `Uint8Array` that have their
 * prototype changed to `Buffer.prototype`. Furthermore, `Buffer` is a subclass of
 * `Uint8Array`, so the returned instances will have all the node `Buffer` methods
 * and the `Uint8Array` methods. Square bracket notation works as expected -- it
 * returns a single octet.
 *
 * The `Uint8Array` prototype remains unmodified.
 */

function Buffer (arg, encodingOrOffset, length) {
  if (!Buffer.TYPED_ARRAY_SUPPORT && !(this instanceof Buffer)) {
    return new Buffer(arg, encodingOrOffset, length)
  }

  // Common case.
  if (typeof arg === 'number') {
    if (typeof encodingOrOffset === 'string') {
      throw new Error(
        'If encoding is specified then the first argument must be a string'
      )
    }
    return allocUnsafe(this, arg)
  }
  return from(this, arg, encodingOrOffset, length)
}

Buffer.poolSize = 8192 // not used by this implementation

// TODO: Legacy, not needed anymore. Remove in next major version.
Buffer._augment = function (arr) {
  arr.__proto__ = Buffer.prototype
  return arr
}

function from (that, value, encodingOrOffset, length) {
  if (typeof value === 'number') {
    throw new TypeError('"value" argument must not be a number')
  }

  if (typeof ArrayBuffer !== 'undefined' && value instanceof ArrayBuffer) {
    return fromArrayBuffer(that, value, encodingOrOffset, length)
  }

  if (typeof value === 'string') {
    return fromString(that, value, encodingOrOffset)
  }

  return fromObject(that, value)
}

/**
 * Functionally equivalent to Buffer(arg, encoding) but throws a TypeError
 * if value is a number.
 * Buffer.from(str[, encoding])
 * Buffer.from(array)
 * Buffer.from(buffer)
 * Buffer.from(arrayBuffer[, byteOffset[, length]])
 **/
Buffer.from = function (value, encodingOrOffset, length) {
  return from(null, value, encodingOrOffset, length)
}

if (Buffer.TYPED_ARRAY_SUPPORT) {
  Buffer.prototype.__proto__ = Uint8Array.prototype
  Buffer.__proto__ = Uint8Array
  if (typeof Symbol !== 'undefined' && Symbol.species &&
      Buffer[Symbol.species] === Buffer) {
    // Fix subarray() in ES2016. See: https://github.com/feross/buffer/pull/97
    Object.defineProperty(Buffer, Symbol.species, {
      value: null,
      configurable: true
    })
  }
}

function assertSize (size) {
  if (typeof size !== 'number') {
    throw new TypeError('"size" argument must be a number')
  } else if (size < 0) {
    throw new RangeError('"size" argument must not be negative')
  }
}

function alloc (that, size, fill, encoding) {
  assertSize(size)
  if (size <= 0) {
    return createBuffer(that, size)
  }
  if (fill !== undefined) {
    // Only pay attention to encoding if it's a string. This
    // prevents accidentally sending in a number that would
    // be interpretted as a start offset.
    return typeof encoding === 'string'
      ? createBuffer(that, size).fill(fill, encoding)
      : createBuffer(that, size).fill(fill)
  }
  return createBuffer(that, size)
}

/**
 * Creates a new filled Buffer instance.
 * alloc(size[, fill[, encoding]])
 **/
Buffer.alloc = function (size, fill, encoding) {
  return alloc(null, size, fill, encoding)
}

function allocUnsafe (that, size) {
  assertSize(size)
  that = createBuffer(that, size < 0 ? 0 : checked(size) | 0)
  if (!Buffer.TYPED_ARRAY_SUPPORT) {
    for (var i = 0; i < size; ++i) {
      that[i] = 0
    }
  }
  return that
}

/**
 * Equivalent to Buffer(num), by default creates a non-zero-filled Buffer instance.
 * */
Buffer.allocUnsafe = function (size) {
  return allocUnsafe(null, size)
}
/**
 * Equivalent to SlowBuffer(num), by default creates a non-zero-filled Buffer instance.
 */
Buffer.allocUnsafeSlow = function (size) {
  return allocUnsafe(null, size)
}

function fromString (that, string, encoding) {
  if (typeof encoding !== 'string' || encoding === '') {
    encoding = 'utf8'
  }

  if (!Buffer.isEncoding(encoding)) {
    throw new TypeError('"encoding" must be a valid string encoding')
  }

  var length = byteLength(string, encoding) | 0
  that = createBuffer(that, length)

  var actual = that.write(string, encoding)

  if (actual !== length) {
    // Writing a hex string, for example, that contains invalid characters will
    // cause everything after the first invalid character to be ignored. (e.g.
    // 'abxxcd' will be treated as 'ab')
    that = that.slice(0, actual)
  }

  return that
}

function fromArrayLike (that, array) {
  var length = array.length < 0 ? 0 : checked(array.length) | 0
  that = createBuffer(that, length)
  for (var i = 0; i < length; i += 1) {
    that[i] = array[i] & 255
  }
  return that
}

function fromArrayBuffer (that, array, byteOffset, length) {
  array.byteLength // this throws if `array` is not a valid ArrayBuffer

  if (byteOffset < 0 || array.byteLength < byteOffset) {
    throw new RangeError('\'offset\' is out of bounds')
  }

  if (array.byteLength < byteOffset + (length || 0)) {
    throw new RangeError('\'length\' is out of bounds')
  }

  if (byteOffset === undefined && length === undefined) {
    array = new Uint8Array(array)
  } else if (length === undefined) {
    array = new Uint8Array(array, byteOffset)
  } else {
    array = new Uint8Array(array, byteOffset, length)
  }

  if (Buffer.TYPED_ARRAY_SUPPORT) {
    // Return an augmented `Uint8Array` instance, for best performance
    that = array
    that.__proto__ = Buffer.prototype
  } else {
    // Fallback: Return an object instance of the Buffer class
    that = fromArrayLike(that, array)
  }
  return that
}

function fromObject (that, obj) {
  if (Buffer.isBuffer(obj)) {
    var len = checked(obj.length) | 0
    that = createBuffer(that, len)

    if (that.length === 0) {
      return that
    }

    obj.copy(that, 0, 0, len)
    return that
  }

  if (obj) {
    if ((typeof ArrayBuffer !== 'undefined' &&
        obj.buffer instanceof ArrayBuffer) || 'length' in obj) {
      if (typeof obj.length !== 'number' || isnan(obj.length)) {
        return createBuffer(that, 0)
      }
      return fromArrayLike(that, obj)
    }

    if (obj.type === 'Buffer' && isArray(obj.data)) {
      return fromArrayLike(that, obj.data)
    }
  }

  throw new TypeError('First argument must be a string, Buffer, ArrayBuffer, Array, or array-like object.')
}

function checked (length) {
  // Note: cannot use `length < kMaxLength()` here because that fails when
  // length is NaN (which is otherwise coerced to zero.)
  if (length >= kMaxLength()) {
    throw new RangeError('Attempt to allocate Buffer larger than maximum ' +
                         'size: 0x' + kMaxLength().toString(16) + ' bytes')
  }
  return length | 0
}

function SlowBuffer (length) {
  if (+length != length) { // eslint-disable-line eqeqeq
    length = 0
  }
  return Buffer.alloc(+length)
}

Buffer.isBuffer = function isBuffer (b) {
  return !!(b != null && b._isBuffer)
}

Buffer.compare = function compare (a, b) {
  if (!Buffer.isBuffer(a) || !Buffer.isBuffer(b)) {
    throw new TypeError('Arguments must be Buffers')
  }

  if (a === b) return 0

  var x = a.length
  var y = b.length

  for (var i = 0, len = Math.min(x, y); i < len; ++i) {
    if (a[i] !== b[i]) {
      x = a[i]
      y = b[i]
      break
    }
  }

  if (x < y) return -1
  if (y < x) return 1
  return 0
}

Buffer.isEncoding = function isEncoding (encoding) {
  switch (String(encoding).toLowerCase()) {
    case 'hex':
    case 'utf8':
    case 'utf-8':
    case 'ascii':
    case 'latin1':
    case 'binary':
    case 'base64':
    case 'ucs2':
    case 'ucs-2':
    case 'utf16le':
    case 'utf-16le':
      return true
    default:
      return false
  }
}

Buffer.concat = function concat (list, length) {
  if (!isArray(list)) {
    throw new TypeError('"list" argument must be an Array of Buffers')
  }

  if (list.length === 0) {
    return Buffer.alloc(0)
  }

  var i
  if (length === undefined) {
    length = 0
    for (i = 0; i < list.length; ++i) {
      length += list[i].length
    }
  }

  var buffer = Buffer.allocUnsafe(length)
  var pos = 0
  for (i = 0; i < list.length; ++i) {
    var buf = list[i]
    if (!Buffer.isBuffer(buf)) {
      throw new TypeError('"list" argument must be an Array of Buffers')
    }
    buf.copy(buffer, pos)
    pos += buf.length
  }
  return buffer
}

function byteLength (string, encoding) {
  if (Buffer.isBuffer(string)) {
    return string.length
  }
  if (typeof ArrayBuffer !== 'undefined' && typeof ArrayBuffer.isView === 'function' &&
      (ArrayBuffer.isView(string) || string instanceof ArrayBuffer)) {
    return string.byteLength
  }
  if (typeof string !== 'string') {
    string = '' + string
  }

  var len = string.length
  if (len === 0) return 0

  // Use a for loop to avoid recursion
  var loweredCase = false
  for (;;) {
    switch (encoding) {
      case 'ascii':
      case 'latin1':
      case 'binary':
        return len
      case 'utf8':
      case 'utf-8':
      case undefined:
        return utf8ToBytes(string).length
      case 'ucs2':
      case 'ucs-2':
      case 'utf16le':
      case 'utf-16le':
        return len * 2
      case 'hex':
        return len >>> 1
      case 'base64':
        return base64ToBytes(string).length
      default:
        if (loweredCase) return utf8ToBytes(string).length // assume utf8
        encoding = ('' + encoding).toLowerCase()
        loweredCase = true
    }
  }
}
Buffer.byteLength = byteLength

function slowToString (encoding, start, end) {
  var loweredCase = false

  // No need to verify that "this.length <= MAX_UINT32" since it's a read-only
  // property of a typed array.

  // This behaves neither like String nor Uint8Array in that we set start/end
  // to their upper/lower bounds if the value passed is out of range.
  // undefined is handled specially as per ECMA-262 6th Edition,
  // Section 13.3.3.7 Runtime Semantics: KeyedBindingInitialization.
  if (start === undefined || start < 0) {
    start = 0
  }
  // Return early if start > this.length. Done here to prevent potential uint32
  // coercion fail below.
  if (start > this.length) {
    return ''
  }

  if (end === undefined || end > this.length) {
    end = this.length
  }

  if (end <= 0) {
    return ''
  }

  // Force coersion to uint32. This will also coerce falsey/NaN values to 0.
  end >>>= 0
  start >>>= 0

  if (end <= start) {
    return ''
  }

  if (!encoding) encoding = 'utf8'

  while (true) {
    switch (encoding) {
      case 'hex':
        return hexSlice(this, start, end)

      case 'utf8':
      case 'utf-8':
        return utf8Slice(this, start, end)

      case 'ascii':
        return asciiSlice(this, start, end)

      case 'latin1':
      case 'binary':
        return latin1Slice(this, start, end)

      case 'base64':
        return base64Slice(this, start, end)

      case 'ucs2':
      case 'ucs-2':
      case 'utf16le':
      case 'utf-16le':
        return utf16leSlice(this, start, end)

      default:
        if (loweredCase) throw new TypeError('Unknown encoding: ' + encoding)
        encoding = (encoding + '').toLowerCase()
        loweredCase = true
    }
  }
}

// The property is used by `Buffer.isBuffer` and `is-buffer` (in Safari 5-7) to detect
// Buffer instances.
Buffer.prototype._isBuffer = true

function swap (b, n, m) {
  var i = b[n]
  b[n] = b[m]
  b[m] = i
}

Buffer.prototype.swap16 = function swap16 () {
  var len = this.length
  if (len % 2 !== 0) {
    throw new RangeError('Buffer size must be a multiple of 16-bits')
  }
  for (var i = 0; i < len; i += 2) {
    swap(this, i, i + 1)
  }
  return this
}

Buffer.prototype.swap32 = function swap32 () {
  var len = this.length
  if (len % 4 !== 0) {
    throw new RangeError('Buffer size must be a multiple of 32-bits')
  }
  for (var i = 0; i < len; i += 4) {
    swap(this, i, i + 3)
    swap(this, i + 1, i + 2)
  }
  return this
}

Buffer.prototype.swap64 = function swap64 () {
  var len = this.length
  if (len % 8 !== 0) {
    throw new RangeError('Buffer size must be a multiple of 64-bits')
  }
  for (var i = 0; i < len; i += 8) {
    swap(this, i, i + 7)
    swap(this, i + 1, i + 6)
    swap(this, i + 2, i + 5)
    swap(this, i + 3, i + 4)
  }
  return this
}

Buffer.prototype.toString = function toString () {
  var length = this.length | 0
  if (length === 0) return ''
  if (arguments.length === 0) return utf8Slice(this, 0, length)
  return slowToString.apply(this, arguments)
}

Buffer.prototype.equals = function equals (b) {
  if (!Buffer.isBuffer(b)) throw new TypeError('Argument must be a Buffer')
  if (this === b) return true
  return Buffer.compare(this, b) === 0
}

Buffer.prototype.inspect = function inspect () {
  var str = ''
  var max = exports.INSPECT_MAX_BYTES
  if (this.length > 0) {
    str = this.toString('hex', 0, max).match(/.{2}/g).join(' ')
    if (this.length > max) str += ' ... '
  }
  return '<Buffer ' + str + '>'
}

Buffer.prototype.compare = function compare (target, start, end, thisStart, thisEnd) {
  if (!Buffer.isBuffer(target)) {
    throw new TypeError('Argument must be a Buffer')
  }

  if (start === undefined) {
    start = 0
  }
  if (end === undefined) {
    end = target ? target.length : 0
  }
  if (thisStart === undefined) {
    thisStart = 0
  }
  if (thisEnd === undefined) {
    thisEnd = this.length
  }

  if (start < 0 || end > target.length || thisStart < 0 || thisEnd > this.length) {
    throw new RangeError('out of range index')
  }

  if (thisStart >= thisEnd && start >= end) {
    return 0
  }
  if (thisStart >= thisEnd) {
    return -1
  }
  if (start >= end) {
    return 1
  }

  start >>>= 0
  end >>>= 0
  thisStart >>>= 0
  thisEnd >>>= 0

  if (this === target) return 0

  var x = thisEnd - thisStart
  var y = end - start
  var len = Math.min(x, y)

  var thisCopy = this.slice(thisStart, thisEnd)
  var targetCopy = target.slice(start, end)

  for (var i = 0; i < len; ++i) {
    if (thisCopy[i] !== targetCopy[i]) {
      x = thisCopy[i]
      y = targetCopy[i]
      break
    }
  }

  if (x < y) return -1
  if (y < x) return 1
  return 0
}

// Finds either the first index of `val` in `buffer` at offset >= `byteOffset`,
// OR the last index of `val` in `buffer` at offset <= `byteOffset`.
//
// Arguments:
// - buffer - a Buffer to search
// - val - a string, Buffer, or number
// - byteOffset - an index into `buffer`; will be clamped to an int32
// - encoding - an optional encoding, relevant is val is a string
// - dir - true for indexOf, false for lastIndexOf
function bidirectionalIndexOf (buffer, val, byteOffset, encoding, dir) {
  // Empty buffer means no match
  if (buffer.length === 0) return -1

  // Normalize byteOffset
  if (typeof byteOffset === 'string') {
    encoding = byteOffset
    byteOffset = 0
  } else if (byteOffset > 0x7fffffff) {
    byteOffset = 0x7fffffff
  } else if (byteOffset < -0x80000000) {
    byteOffset = -0x80000000
  }
  byteOffset = +byteOffset  // Coerce to Number.
  if (isNaN(byteOffset)) {
    // byteOffset: it it's undefined, null, NaN, "foo", etc, search whole buffer
    byteOffset = dir ? 0 : (buffer.length - 1)
  }

  // Normalize byteOffset: negative offsets start from the end of the buffer
  if (byteOffset < 0) byteOffset = buffer.length + byteOffset
  if (byteOffset >= buffer.length) {
    if (dir) return -1
    else byteOffset = buffer.length - 1
  } else if (byteOffset < 0) {
    if (dir) byteOffset = 0
    else return -1
  }

  // Normalize val
  if (typeof val === 'string') {
    val = Buffer.from(val, encoding)
  }

  // Finally, search either indexOf (if dir is true) or lastIndexOf
  if (Buffer.isBuffer(val)) {
    // Special case: looking for empty string/buffer always fails
    if (val.length === 0) {
      return -1
    }
    return arrayIndexOf(buffer, val, byteOffset, encoding, dir)
  } else if (typeof val === 'number') {
    val = val & 0xFF // Search for a byte value [0-255]
    if (Buffer.TYPED_ARRAY_SUPPORT &&
        typeof Uint8Array.prototype.indexOf === 'function') {
      if (dir) {
        return Uint8Array.prototype.indexOf.call(buffer, val, byteOffset)
      } else {
        return Uint8Array.prototype.lastIndexOf.call(buffer, val, byteOffset)
      }
    }
    return arrayIndexOf(buffer, [ val ], byteOffset, encoding, dir)
  }

  throw new TypeError('val must be string, number or Buffer')
}

function arrayIndexOf (arr, val, byteOffset, encoding, dir) {
  var indexSize = 1
  var arrLength = arr.length
  var valLength = val.length

  if (encoding !== undefined) {
    encoding = String(encoding).toLowerCase()
    if (encoding === 'ucs2' || encoding === 'ucs-2' ||
        encoding === 'utf16le' || encoding === 'utf-16le') {
      if (arr.length < 2 || val.length < 2) {
        return -1
      }
      indexSize = 2
      arrLength /= 2
      valLength /= 2
      byteOffset /= 2
    }
  }

  function read (buf, i) {
    if (indexSize === 1) {
      return buf[i]
    } else {
      return buf.readUInt16BE(i * indexSize)
    }
  }

  var i
  if (dir) {
    var foundIndex = -1
    for (i = byteOffset; i < arrLength; i++) {
      if (read(arr, i) === read(val, foundIndex === -1 ? 0 : i - foundIndex)) {
        if (foundIndex === -1) foundIndex = i
        if (i - foundIndex + 1 === valLength) return foundIndex * indexSize
      } else {
        if (foundIndex !== -1) i -= i - foundIndex
        foundIndex = -1
      }
    }
  } else {
    if (byteOffset + valLength > arrLength) byteOffset = arrLength - valLength
    for (i = byteOffset; i >= 0; i--) {
      var found = true
      for (var j = 0; j < valLength; j++) {
        if (read(arr, i + j) !== read(val, j)) {
          found = false
          break
        }
      }
      if (found) return i
    }
  }

  return -1
}

Buffer.prototype.includes = function includes (val, byteOffset, encoding) {
  return this.indexOf(val, byteOffset, encoding) !== -1
}

Buffer.prototype.indexOf = function indexOf (val, byteOffset, encoding) {
  return bidirectionalIndexOf(this, val, byteOffset, encoding, true)
}

Buffer.prototype.lastIndexOf = function lastIndexOf (val, byteOffset, encoding) {
  return bidirectionalIndexOf(this, val, byteOffset, encoding, false)
}

function hexWrite (buf, string, offset, length) {
  offset = Number(offset) || 0
  var remaining = buf.length - offset
  if (!length) {
    length = remaining
  } else {
    length = Number(length)
    if (length > remaining) {
      length = remaining
    }
  }

  // must be an even number of digits
  var strLen = string.length
  if (strLen % 2 !== 0) throw new TypeError('Invalid hex string')

  if (length > strLen / 2) {
    length = strLen / 2
  }
  for (var i = 0; i < length; ++i) {
    var parsed = parseInt(string.substr(i * 2, 2), 16)
    if (isNaN(parsed)) return i
    buf[offset + i] = parsed
  }
  return i
}

function utf8Write (buf, string, offset, length) {
  return blitBuffer(utf8ToBytes(string, buf.length - offset), buf, offset, length)
}

function asciiWrite (buf, string, offset, length) {
  return blitBuffer(asciiToBytes(string), buf, offset, length)
}

function latin1Write (buf, string, offset, length) {
  return asciiWrite(buf, string, offset, length)
}

function base64Write (buf, string, offset, length) {
  return blitBuffer(base64ToBytes(string), buf, offset, length)
}

function ucs2Write (buf, string, offset, length) {
  return blitBuffer(utf16leToBytes(string, buf.length - offset), buf, offset, length)
}

Buffer.prototype.write = function write (string, offset, length, encoding) {
  // Buffer#write(string)
  if (offset === undefined) {
    encoding = 'utf8'
    length = this.length
    offset = 0
  // Buffer#write(string, encoding)
  } else if (length === undefined && typeof offset === 'string') {
    encoding = offset
    length = this.length
    offset = 0
  // Buffer#write(string, offset[, length][, encoding])
  } else if (isFinite(offset)) {
    offset = offset | 0
    if (isFinite(length)) {
      length = length | 0
      if (encoding === undefined) encoding = 'utf8'
    } else {
      encoding = length
      length = undefined
    }
  // legacy write(string, encoding, offset, length) - remove in v0.13
  } else {
    throw new Error(
      'Buffer.write(string, encoding, offset[, length]) is no longer supported'
    )
  }

  var remaining = this.length - offset
  if (length === undefined || length > remaining) length = remaining

  if ((string.length > 0 && (length < 0 || offset < 0)) || offset > this.length) {
    throw new RangeError('Attempt to write outside buffer bounds')
  }

  if (!encoding) encoding = 'utf8'

  var loweredCase = false
  for (;;) {
    switch (encoding) {
      case 'hex':
        return hexWrite(this, string, offset, length)

      case 'utf8':
      case 'utf-8':
        return utf8Write(this, string, offset, length)

      case 'ascii':
        return asciiWrite(this, string, offset, length)

      case 'latin1':
      case 'binary':
        return latin1Write(this, string, offset, length)

      case 'base64':
        // Warning: maxLength not taken into account in base64Write
        return base64Write(this, string, offset, length)

      case 'ucs2':
      case 'ucs-2':
      case 'utf16le':
      case 'utf-16le':
        return ucs2Write(this, string, offset, length)

      default:
        if (loweredCase) throw new TypeError('Unknown encoding: ' + encoding)
        encoding = ('' + encoding).toLowerCase()
        loweredCase = true
    }
  }
}

Buffer.prototype.toJSON = function toJSON () {
  return {
    type: 'Buffer',
    data: Array.prototype.slice.call(this._arr || this, 0)
  }
}

function base64Slice (buf, start, end) {
  if (start === 0 && end === buf.length) {
    return base64.fromByteArray(buf)
  } else {
    return base64.fromByteArray(buf.slice(start, end))
  }
}

function utf8Slice (buf, start, end) {
  end = Math.min(buf.length, end)
  var res = []

  var i = start
  while (i < end) {
    var firstByte = buf[i]
    var codePoint = null
    var bytesPerSequence = (firstByte > 0xEF) ? 4
      : (firstByte > 0xDF) ? 3
      : (firstByte > 0xBF) ? 2
      : 1

    if (i + bytesPerSequence <= end) {
      var secondByte, thirdByte, fourthByte, tempCodePoint

      switch (bytesPerSequence) {
        case 1:
          if (firstByte < 0x80) {
            codePoint = firstByte
          }
          break
        case 2:
          secondByte = buf[i + 1]
          if ((secondByte & 0xC0) === 0x80) {
            tempCodePoint = (firstByte & 0x1F) << 0x6 | (secondByte & 0x3F)
            if (tempCodePoint > 0x7F) {
              codePoint = tempCodePoint
            }
          }
          break
        case 3:
          secondByte = buf[i + 1]
          thirdByte = buf[i + 2]
          if ((secondByte & 0xC0) === 0x80 && (thirdByte & 0xC0) === 0x80) {
            tempCodePoint = (firstByte & 0xF) << 0xC | (secondByte & 0x3F) << 0x6 | (thirdByte & 0x3F)
            if (tempCodePoint > 0x7FF && (tempCodePoint < 0xD800 || tempCodePoint > 0xDFFF)) {
              codePoint = tempCodePoint
            }
          }
          break
        case 4:
          secondByte = buf[i + 1]
          thirdByte = buf[i + 2]
          fourthByte = buf[i + 3]
          if ((secondByte & 0xC0) === 0x80 && (thirdByte & 0xC0) === 0x80 && (fourthByte & 0xC0) === 0x80) {
            tempCodePoint = (firstByte & 0xF) << 0x12 | (secondByte & 0x3F) << 0xC | (thirdByte & 0x3F) << 0x6 | (fourthByte & 0x3F)
            if (tempCodePoint > 0xFFFF && tempCodePoint < 0x110000) {
              codePoint = tempCodePoint
            }
          }
      }
    }

    if (codePoint === null) {
      // we did not generate a valid codePoint so insert a
      // replacement char (U+FFFD) and advance only 1 byte
      codePoint = 0xFFFD
      bytesPerSequence = 1
    } else if (codePoint > 0xFFFF) {
      // encode to utf16 (surrogate pair dance)
      codePoint -= 0x10000
      res.push(codePoint >>> 10 & 0x3FF | 0xD800)
      codePoint = 0xDC00 | codePoint & 0x3FF
    }

    res.push(codePoint)
    i += bytesPerSequence
  }

  return decodeCodePointsArray(res)
}

// Based on http://stackoverflow.com/a/22747272/680742, the browser with
// the lowest limit is Chrome, with 0x10000 args.
// We go 1 magnitude less, for safety
var MAX_ARGUMENTS_LENGTH = 0x1000

function decodeCodePointsArray (codePoints) {
  var len = codePoints.length
  if (len <= MAX_ARGUMENTS_LENGTH) {
    return String.fromCharCode.apply(String, codePoints) // avoid extra slice()
  }

  // Decode in chunks to avoid "call stack size exceeded".
  var res = ''
  var i = 0
  while (i < len) {
    res += String.fromCharCode.apply(
      String,
      codePoints.slice(i, i += MAX_ARGUMENTS_LENGTH)
    )
  }
  return res
}

function asciiSlice (buf, start, end) {
  var ret = ''
  end = Math.min(buf.length, end)

  for (var i = start; i < end; ++i) {
    ret += String.fromCharCode(buf[i] & 0x7F)
  }
  return ret
}

function latin1Slice (buf, start, end) {
  var ret = ''
  end = Math.min(buf.length, end)

  for (var i = start; i < end; ++i) {
    ret += String.fromCharCode(buf[i])
  }
  return ret
}

function hexSlice (buf, start, end) {
  var len = buf.length

  if (!start || start < 0) start = 0
  if (!end || end < 0 || end > len) end = len

  var out = ''
  for (var i = start; i < end; ++i) {
    out += toHex(buf[i])
  }
  return out
}

function utf16leSlice (buf, start, end) {
  var bytes = buf.slice(start, end)
  var res = ''
  for (var i = 0; i < bytes.length; i += 2) {
    res += String.fromCharCode(bytes[i] + bytes[i + 1] * 256)
  }
  return res
}

Buffer.prototype.slice = function slice (start, end) {
  var len = this.length
  start = ~~start
  end = end === undefined ? len : ~~end

  if (start < 0) {
    start += len
    if (start < 0) start = 0
  } else if (start > len) {
    start = len
  }

  if (end < 0) {
    end += len
    if (end < 0) end = 0
  } else if (end > len) {
    end = len
  }

  if (end < start) end = start

  var newBuf
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    newBuf = this.subarray(start, end)
    newBuf.__proto__ = Buffer.prototype
  } else {
    var sliceLen = end - start
    newBuf = new Buffer(sliceLen, undefined)
    for (var i = 0; i < sliceLen; ++i) {
      newBuf[i] = this[i + start]
    }
  }

  return newBuf
}

/*
 * Need to make sure that buffer isn't trying to write out of bounds.
 */
function checkOffset (offset, ext, length) {
  if ((offset % 1) !== 0 || offset < 0) throw new RangeError('offset is not uint')
  if (offset + ext > length) throw new RangeError('Trying to access beyond buffer length')
}

Buffer.prototype.readUIntLE = function readUIntLE (offset, byteLength, noAssert) {
  offset = offset | 0
  byteLength = byteLength | 0
  if (!noAssert) checkOffset(offset, byteLength, this.length)

  var val = this[offset]
  var mul = 1
  var i = 0
  while (++i < byteLength && (mul *= 0x100)) {
    val += this[offset + i] * mul
  }

  return val
}

Buffer.prototype.readUIntBE = function readUIntBE (offset, byteLength, noAssert) {
  offset = offset | 0
  byteLength = byteLength | 0
  if (!noAssert) {
    checkOffset(offset, byteLength, this.length)
  }

  var val = this[offset + --byteLength]
  var mul = 1
  while (byteLength > 0 && (mul *= 0x100)) {
    val += this[offset + --byteLength] * mul
  }

  return val
}

Buffer.prototype.readUInt8 = function readUInt8 (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 1, this.length)
  return this[offset]
}

Buffer.prototype.readUInt16LE = function readUInt16LE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 2, this.length)
  return this[offset] | (this[offset + 1] << 8)
}

Buffer.prototype.readUInt16BE = function readUInt16BE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 2, this.length)
  return (this[offset] << 8) | this[offset + 1]
}

Buffer.prototype.readUInt32LE = function readUInt32LE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length)

  return ((this[offset]) |
      (this[offset + 1] << 8) |
      (this[offset + 2] << 16)) +
      (this[offset + 3] * 0x1000000)
}

Buffer.prototype.readUInt32BE = function readUInt32BE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length)

  return (this[offset] * 0x1000000) +
    ((this[offset + 1] << 16) |
    (this[offset + 2] << 8) |
    this[offset + 3])
}

Buffer.prototype.readIntLE = function readIntLE (offset, byteLength, noAssert) {
  offset = offset | 0
  byteLength = byteLength | 0
  if (!noAssert) checkOffset(offset, byteLength, this.length)

  var val = this[offset]
  var mul = 1
  var i = 0
  while (++i < byteLength && (mul *= 0x100)) {
    val += this[offset + i] * mul
  }
  mul *= 0x80

  if (val >= mul) val -= Math.pow(2, 8 * byteLength)

  return val
}

Buffer.prototype.readIntBE = function readIntBE (offset, byteLength, noAssert) {
  offset = offset | 0
  byteLength = byteLength | 0
  if (!noAssert) checkOffset(offset, byteLength, this.length)

  var i = byteLength
  var mul = 1
  var val = this[offset + --i]
  while (i > 0 && (mul *= 0x100)) {
    val += this[offset + --i] * mul
  }
  mul *= 0x80

  if (val >= mul) val -= Math.pow(2, 8 * byteLength)

  return val
}

Buffer.prototype.readInt8 = function readInt8 (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 1, this.length)
  if (!(this[offset] & 0x80)) return (this[offset])
  return ((0xff - this[offset] + 1) * -1)
}

Buffer.prototype.readInt16LE = function readInt16LE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 2, this.length)
  var val = this[offset] | (this[offset + 1] << 8)
  return (val & 0x8000) ? val | 0xFFFF0000 : val
}

Buffer.prototype.readInt16BE = function readInt16BE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 2, this.length)
  var val = this[offset + 1] | (this[offset] << 8)
  return (val & 0x8000) ? val | 0xFFFF0000 : val
}

Buffer.prototype.readInt32LE = function readInt32LE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length)

  return (this[offset]) |
    (this[offset + 1] << 8) |
    (this[offset + 2] << 16) |
    (this[offset + 3] << 24)
}

Buffer.prototype.readInt32BE = function readInt32BE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length)

  return (this[offset] << 24) |
    (this[offset + 1] << 16) |
    (this[offset + 2] << 8) |
    (this[offset + 3])
}

Buffer.prototype.readFloatLE = function readFloatLE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length)
  return ieee754.read(this, offset, true, 23, 4)
}

Buffer.prototype.readFloatBE = function readFloatBE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length)
  return ieee754.read(this, offset, false, 23, 4)
}

Buffer.prototype.readDoubleLE = function readDoubleLE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 8, this.length)
  return ieee754.read(this, offset, true, 52, 8)
}

Buffer.prototype.readDoubleBE = function readDoubleBE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 8, this.length)
  return ieee754.read(this, offset, false, 52, 8)
}

function checkInt (buf, value, offset, ext, max, min) {
  if (!Buffer.isBuffer(buf)) throw new TypeError('"buffer" argument must be a Buffer instance')
  if (value > max || value < min) throw new RangeError('"value" argument is out of bounds')
  if (offset + ext > buf.length) throw new RangeError('Index out of range')
}

Buffer.prototype.writeUIntLE = function writeUIntLE (value, offset, byteLength, noAssert) {
  value = +value
  offset = offset | 0
  byteLength = byteLength | 0
  if (!noAssert) {
    var maxBytes = Math.pow(2, 8 * byteLength) - 1
    checkInt(this, value, offset, byteLength, maxBytes, 0)
  }

  var mul = 1
  var i = 0
  this[offset] = value & 0xFF
  while (++i < byteLength && (mul *= 0x100)) {
    this[offset + i] = (value / mul) & 0xFF
  }

  return offset + byteLength
}

Buffer.prototype.writeUIntBE = function writeUIntBE (value, offset, byteLength, noAssert) {
  value = +value
  offset = offset | 0
  byteLength = byteLength | 0
  if (!noAssert) {
    var maxBytes = Math.pow(2, 8 * byteLength) - 1
    checkInt(this, value, offset, byteLength, maxBytes, 0)
  }

  var i = byteLength - 1
  var mul = 1
  this[offset + i] = value & 0xFF
  while (--i >= 0 && (mul *= 0x100)) {
    this[offset + i] = (value / mul) & 0xFF
  }

  return offset + byteLength
}

Buffer.prototype.writeUInt8 = function writeUInt8 (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 1, 0xff, 0)
  if (!Buffer.TYPED_ARRAY_SUPPORT) value = Math.floor(value)
  this[offset] = (value & 0xff)
  return offset + 1
}

function objectWriteUInt16 (buf, value, offset, littleEndian) {
  if (value < 0) value = 0xffff + value + 1
  for (var i = 0, j = Math.min(buf.length - offset, 2); i < j; ++i) {
    buf[offset + i] = (value & (0xff << (8 * (littleEndian ? i : 1 - i)))) >>>
      (littleEndian ? i : 1 - i) * 8
  }
}

Buffer.prototype.writeUInt16LE = function writeUInt16LE (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 2, 0xffff, 0)
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value & 0xff)
    this[offset + 1] = (value >>> 8)
  } else {
    objectWriteUInt16(this, value, offset, true)
  }
  return offset + 2
}

Buffer.prototype.writeUInt16BE = function writeUInt16BE (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 2, 0xffff, 0)
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value >>> 8)
    this[offset + 1] = (value & 0xff)
  } else {
    objectWriteUInt16(this, value, offset, false)
  }
  return offset + 2
}

function objectWriteUInt32 (buf, value, offset, littleEndian) {
  if (value < 0) value = 0xffffffff + value + 1
  for (var i = 0, j = Math.min(buf.length - offset, 4); i < j; ++i) {
    buf[offset + i] = (value >>> (littleEndian ? i : 3 - i) * 8) & 0xff
  }
}

Buffer.prototype.writeUInt32LE = function writeUInt32LE (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 4, 0xffffffff, 0)
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset + 3] = (value >>> 24)
    this[offset + 2] = (value >>> 16)
    this[offset + 1] = (value >>> 8)
    this[offset] = (value & 0xff)
  } else {
    objectWriteUInt32(this, value, offset, true)
  }
  return offset + 4
}

Buffer.prototype.writeUInt32BE = function writeUInt32BE (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 4, 0xffffffff, 0)
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value >>> 24)
    this[offset + 1] = (value >>> 16)
    this[offset + 2] = (value >>> 8)
    this[offset + 3] = (value & 0xff)
  } else {
    objectWriteUInt32(this, value, offset, false)
  }
  return offset + 4
}

Buffer.prototype.writeIntLE = function writeIntLE (value, offset, byteLength, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) {
    var limit = Math.pow(2, 8 * byteLength - 1)

    checkInt(this, value, offset, byteLength, limit - 1, -limit)
  }

  var i = 0
  var mul = 1
  var sub = 0
  this[offset] = value & 0xFF
  while (++i < byteLength && (mul *= 0x100)) {
    if (value < 0 && sub === 0 && this[offset + i - 1] !== 0) {
      sub = 1
    }
    this[offset + i] = ((value / mul) >> 0) - sub & 0xFF
  }

  return offset + byteLength
}

Buffer.prototype.writeIntBE = function writeIntBE (value, offset, byteLength, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) {
    var limit = Math.pow(2, 8 * byteLength - 1)

    checkInt(this, value, offset, byteLength, limit - 1, -limit)
  }

  var i = byteLength - 1
  var mul = 1
  var sub = 0
  this[offset + i] = value & 0xFF
  while (--i >= 0 && (mul *= 0x100)) {
    if (value < 0 && sub === 0 && this[offset + i + 1] !== 0) {
      sub = 1
    }
    this[offset + i] = ((value / mul) >> 0) - sub & 0xFF
  }

  return offset + byteLength
}

Buffer.prototype.writeInt8 = function writeInt8 (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 1, 0x7f, -0x80)
  if (!Buffer.TYPED_ARRAY_SUPPORT) value = Math.floor(value)
  if (value < 0) value = 0xff + value + 1
  this[offset] = (value & 0xff)
  return offset + 1
}

Buffer.prototype.writeInt16LE = function writeInt16LE (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 2, 0x7fff, -0x8000)
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value & 0xff)
    this[offset + 1] = (value >>> 8)
  } else {
    objectWriteUInt16(this, value, offset, true)
  }
  return offset + 2
}

Buffer.prototype.writeInt16BE = function writeInt16BE (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 2, 0x7fff, -0x8000)
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value >>> 8)
    this[offset + 1] = (value & 0xff)
  } else {
    objectWriteUInt16(this, value, offset, false)
  }
  return offset + 2
}

Buffer.prototype.writeInt32LE = function writeInt32LE (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 4, 0x7fffffff, -0x80000000)
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value & 0xff)
    this[offset + 1] = (value >>> 8)
    this[offset + 2] = (value >>> 16)
    this[offset + 3] = (value >>> 24)
  } else {
    objectWriteUInt32(this, value, offset, true)
  }
  return offset + 4
}

Buffer.prototype.writeInt32BE = function writeInt32BE (value, offset, noAssert) {
  value = +value
  offset = offset | 0
  if (!noAssert) checkInt(this, value, offset, 4, 0x7fffffff, -0x80000000)
  if (value < 0) value = 0xffffffff + value + 1
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value >>> 24)
    this[offset + 1] = (value >>> 16)
    this[offset + 2] = (value >>> 8)
    this[offset + 3] = (value & 0xff)
  } else {
    objectWriteUInt32(this, value, offset, false)
  }
  return offset + 4
}

function checkIEEE754 (buf, value, offset, ext, max, min) {
  if (offset + ext > buf.length) throw new RangeError('Index out of range')
  if (offset < 0) throw new RangeError('Index out of range')
}

function writeFloat (buf, value, offset, littleEndian, noAssert) {
  if (!noAssert) {
    checkIEEE754(buf, value, offset, 4, 3.4028234663852886e+38, -3.4028234663852886e+38)
  }
  ieee754.write(buf, value, offset, littleEndian, 23, 4)
  return offset + 4
}

Buffer.prototype.writeFloatLE = function writeFloatLE (value, offset, noAssert) {
  return writeFloat(this, value, offset, true, noAssert)
}

Buffer.prototype.writeFloatBE = function writeFloatBE (value, offset, noAssert) {
  return writeFloat(this, value, offset, false, noAssert)
}

function writeDouble (buf, value, offset, littleEndian, noAssert) {
  if (!noAssert) {
    checkIEEE754(buf, value, offset, 8, 1.7976931348623157E+308, -1.7976931348623157E+308)
  }
  ieee754.write(buf, value, offset, littleEndian, 52, 8)
  return offset + 8
}

Buffer.prototype.writeDoubleLE = function writeDoubleLE (value, offset, noAssert) {
  return writeDouble(this, value, offset, true, noAssert)
}

Buffer.prototype.writeDoubleBE = function writeDoubleBE (value, offset, noAssert) {
  return writeDouble(this, value, offset, false, noAssert)
}

// copy(targetBuffer, targetStart=0, sourceStart=0, sourceEnd=buffer.length)
Buffer.prototype.copy = function copy (target, targetStart, start, end) {
  if (!start) start = 0
  if (!end && end !== 0) end = this.length
  if (targetStart >= target.length) targetStart = target.length
  if (!targetStart) targetStart = 0
  if (end > 0 && end < start) end = start

  // Copy 0 bytes; we're done
  if (end === start) return 0
  if (target.length === 0 || this.length === 0) return 0

  // Fatal error conditions
  if (targetStart < 0) {
    throw new RangeError('targetStart out of bounds')
  }
  if (start < 0 || start >= this.length) throw new RangeError('sourceStart out of bounds')
  if (end < 0) throw new RangeError('sourceEnd out of bounds')

  // Are we oob?
  if (end > this.length) end = this.length
  if (target.length - targetStart < end - start) {
    end = target.length - targetStart + start
  }

  var len = end - start
  var i

  if (this === target && start < targetStart && targetStart < end) {
    // descending copy from end
    for (i = len - 1; i >= 0; --i) {
      target[i + targetStart] = this[i + start]
    }
  } else if (len < 1000 || !Buffer.TYPED_ARRAY_SUPPORT) {
    // ascending copy from start
    for (i = 0; i < len; ++i) {
      target[i + targetStart] = this[i + start]
    }
  } else {
    Uint8Array.prototype.set.call(
      target,
      this.subarray(start, start + len),
      targetStart
    )
  }

  return len
}

// Usage:
//    buffer.fill(number[, offset[, end]])
//    buffer.fill(buffer[, offset[, end]])
//    buffer.fill(string[, offset[, end]][, encoding])
Buffer.prototype.fill = function fill (val, start, end, encoding) {
  // Handle string cases:
  if (typeof val === 'string') {
    if (typeof start === 'string') {
      encoding = start
      start = 0
      end = this.length
    } else if (typeof end === 'string') {
      encoding = end
      end = this.length
    }
    if (val.length === 1) {
      var code = val.charCodeAt(0)
      if (code < 256) {
        val = code
      }
    }
    if (encoding !== undefined && typeof encoding !== 'string') {
      throw new TypeError('encoding must be a string')
    }
    if (typeof encoding === 'string' && !Buffer.isEncoding(encoding)) {
      throw new TypeError('Unknown encoding: ' + encoding)
    }
  } else if (typeof val === 'number') {
    val = val & 255
  }

  // Invalid ranges are not set to a default, so can range check early.
  if (start < 0 || this.length < start || this.length < end) {
    throw new RangeError('Out of range index')
  }

  if (end <= start) {
    return this
  }

  start = start >>> 0
  end = end === undefined ? this.length : end >>> 0

  if (!val) val = 0

  var i
  if (typeof val === 'number') {
    for (i = start; i < end; ++i) {
      this[i] = val
    }
  } else {
    var bytes = Buffer.isBuffer(val)
      ? val
      : utf8ToBytes(new Buffer(val, encoding).toString())
    var len = bytes.length
    for (i = 0; i < end - start; ++i) {
      this[i + start] = bytes[i % len]
    }
  }

  return this
}

// HELPER FUNCTIONS
// ================

var INVALID_BASE64_RE = /[^+\/0-9A-Za-z-_]/g

function base64clean (str) {
  // Node strips out invalid characters like \n and \t from the string, base64-js does not
  str = stringtrim(str).replace(INVALID_BASE64_RE, '')
  // Node converts strings with length < 2 to ''
  if (str.length < 2) return ''
  // Node allows for non-padded base64 strings (missing trailing ===), base64-js does not
  while (str.length % 4 !== 0) {
    str = str + '='
  }
  return str
}

function stringtrim (str) {
  if (str.trim) return str.trim()
  return str.replace(/^\s+|\s+$/g, '')
}

function toHex (n) {
  if (n < 16) return '0' + n.toString(16)
  return n.toString(16)
}

function utf8ToBytes (string, units) {
  units = units || Infinity
  var codePoint
  var length = string.length
  var leadSurrogate = null
  var bytes = []

  for (var i = 0; i < length; ++i) {
    codePoint = string.charCodeAt(i)

    // is surrogate component
    if (codePoint > 0xD7FF && codePoint < 0xE000) {
      // last char was a lead
      if (!leadSurrogate) {
        // no lead yet
        if (codePoint > 0xDBFF) {
          // unexpected trail
          if ((units -= 3) > -1) bytes.push(0xEF, 0xBF, 0xBD)
          continue
        } else if (i + 1 === length) {
          // unpaired lead
          if ((units -= 3) > -1) bytes.push(0xEF, 0xBF, 0xBD)
          continue
        }

        // valid lead
        leadSurrogate = codePoint

        continue
      }

      // 2 leads in a row
      if (codePoint < 0xDC00) {
        if ((units -= 3) > -1) bytes.push(0xEF, 0xBF, 0xBD)
        leadSurrogate = codePoint
        continue
      }

      // valid surrogate pair
      codePoint = (leadSurrogate - 0xD800 << 10 | codePoint - 0xDC00) + 0x10000
    } else if (leadSurrogate) {
      // valid bmp char, but last char was a lead
      if ((units -= 3) > -1) bytes.push(0xEF, 0xBF, 0xBD)
    }

    leadSurrogate = null

    // encode utf8
    if (codePoint < 0x80) {
      if ((units -= 1) < 0) break
      bytes.push(codePoint)
    } else if (codePoint < 0x800) {
      if ((units -= 2) < 0) break
      bytes.push(
        codePoint >> 0x6 | 0xC0,
        codePoint & 0x3F | 0x80
      )
    } else if (codePoint < 0x10000) {
      if ((units -= 3) < 0) break
      bytes.push(
        codePoint >> 0xC | 0xE0,
        codePoint >> 0x6 & 0x3F | 0x80,
        codePoint & 0x3F | 0x80
      )
    } else if (codePoint < 0x110000) {
      if ((units -= 4) < 0) break
      bytes.push(
        codePoint >> 0x12 | 0xF0,
        codePoint >> 0xC & 0x3F | 0x80,
        codePoint >> 0x6 & 0x3F | 0x80,
        codePoint & 0x3F | 0x80
      )
    } else {
      throw new Error('Invalid code point')
    }
  }

  return bytes
}

function asciiToBytes (str) {
  var byteArray = []
  for (var i = 0; i < str.length; ++i) {
    // Node's code seems to be doing this and not & 0x7F..
    byteArray.push(str.charCodeAt(i) & 0xFF)
  }
  return byteArray
}

function utf16leToBytes (str, units) {
  var c, hi, lo
  var byteArray = []
  for (var i = 0; i < str.length; ++i) {
    if ((units -= 2) < 0) break

    c = str.charCodeAt(i)
    hi = c >> 8
    lo = c % 256
    byteArray.push(lo)
    byteArray.push(hi)
  }

  return byteArray
}

function base64ToBytes (str) {
  return base64.toByteArray(base64clean(str))
}

function blitBuffer (src, dst, offset, length) {
  for (var i = 0; i < length; ++i) {
    if ((i + offset >= dst.length) || (i >= src.length)) break
    dst[i + offset] = src[i]
  }
  return i
}

function isnan (val) {
  return val !== val // eslint-disable-line no-self-compare
}

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"base64-js":1,"ieee754":10,"isarray":4}],4:[function(require,module,exports){
var toString = {}.toString;

module.exports = Array.isArray || function (arr) {
  return toString.call(arr) == '[object Array]';
};

},{}],5:[function(require,module,exports){
"use strict"

var createThunk = require("./lib/thunk.js")

function Procedure() {
  this.argTypes = []
  this.shimArgs = []
  this.arrayArgs = []
  this.arrayBlockIndices = []
  this.scalarArgs = []
  this.offsetArgs = []
  this.offsetArgIndex = []
  this.indexArgs = []
  this.shapeArgs = []
  this.funcName = ""
  this.pre = null
  this.body = null
  this.post = null
  this.debug = false
}

function compileCwise(user_args) {
  //Create procedure
  var proc = new Procedure()
  
  //Parse blocks
  proc.pre    = user_args.pre
  proc.body   = user_args.body
  proc.post   = user_args.post

  //Parse arguments
  var proc_args = user_args.args.slice(0)
  proc.argTypes = proc_args
  for(var i=0; i<proc_args.length; ++i) {
    var arg_type = proc_args[i]
    if(arg_type === "array" || (typeof arg_type === "object" && arg_type.blockIndices)) {
      proc.argTypes[i] = "array"
      proc.arrayArgs.push(i)
      proc.arrayBlockIndices.push(arg_type.blockIndices ? arg_type.blockIndices : 0)
      proc.shimArgs.push("array" + i)
      if(i < proc.pre.args.length && proc.pre.args[i].count>0) {
        throw new Error("cwise: pre() block may not reference array args")
      }
      if(i < proc.post.args.length && proc.post.args[i].count>0) {
        throw new Error("cwise: post() block may not reference array args")
      }
    } else if(arg_type === "scalar") {
      proc.scalarArgs.push(i)
      proc.shimArgs.push("scalar" + i)
    } else if(arg_type === "index") {
      proc.indexArgs.push(i)
      if(i < proc.pre.args.length && proc.pre.args[i].count > 0) {
        throw new Error("cwise: pre() block may not reference array index")
      }
      if(i < proc.body.args.length && proc.body.args[i].lvalue) {
        throw new Error("cwise: body() block may not write to array index")
      }
      if(i < proc.post.args.length && proc.post.args[i].count > 0) {
        throw new Error("cwise: post() block may not reference array index")
      }
    } else if(arg_type === "shape") {
      proc.shapeArgs.push(i)
      if(i < proc.pre.args.length && proc.pre.args[i].lvalue) {
        throw new Error("cwise: pre() block may not write to array shape")
      }
      if(i < proc.body.args.length && proc.body.args[i].lvalue) {
        throw new Error("cwise: body() block may not write to array shape")
      }
      if(i < proc.post.args.length && proc.post.args[i].lvalue) {
        throw new Error("cwise: post() block may not write to array shape")
      }
    } else if(typeof arg_type === "object" && arg_type.offset) {
      proc.argTypes[i] = "offset"
      proc.offsetArgs.push({ array: arg_type.array, offset:arg_type.offset })
      proc.offsetArgIndex.push(i)
    } else {
      throw new Error("cwise: Unknown argument type " + proc_args[i])
    }
  }
  
  //Make sure at least one array argument was specified
  if(proc.arrayArgs.length <= 0) {
    throw new Error("cwise: No array arguments specified")
  }
  
  //Make sure arguments are correct
  if(proc.pre.args.length > proc_args.length) {
    throw new Error("cwise: Too many arguments in pre() block")
  }
  if(proc.body.args.length > proc_args.length) {
    throw new Error("cwise: Too many arguments in body() block")
  }
  if(proc.post.args.length > proc_args.length) {
    throw new Error("cwise: Too many arguments in post() block")
  }

  //Check debug flag
  proc.debug = !!user_args.printCode || !!user_args.debug
  
  //Retrieve name
  proc.funcName = user_args.funcName || "cwise"
  
  //Read in block size
  proc.blockSize = user_args.blockSize || 64

  return createThunk(proc)
}

module.exports = compileCwise

},{"./lib/thunk.js":7}],6:[function(require,module,exports){
"use strict"

var uniq = require("uniq")

// This function generates very simple loops analogous to how you typically traverse arrays (the outermost loop corresponds to the slowest changing index, the innermost loop to the fastest changing index)
// TODO: If two arrays have the same strides (and offsets) there is potential for decreasing the number of "pointers" and related variables. The drawback is that the type signature would become more specific and that there would thus be less potential for caching, but it might still be worth it, especially when dealing with large numbers of arguments.
function innerFill(order, proc, body) {
  var dimension = order.length
    , nargs = proc.arrayArgs.length
    , has_index = proc.indexArgs.length>0
    , code = []
    , vars = []
    , idx=0, pidx=0, i, j
  for(i=0; i<dimension; ++i) { // Iteration variables
    vars.push(["i",i,"=0"].join(""))
  }
  //Compute scan deltas
  for(j=0; j<nargs; ++j) {
    for(i=0; i<dimension; ++i) {
      pidx = idx
      idx = order[i]
      if(i === 0) { // The innermost/fastest dimension's delta is simply its stride
        vars.push(["d",j,"s",i,"=t",j,"p",idx].join(""))
      } else { // For other dimensions the delta is basically the stride minus something which essentially "rewinds" the previous (more inner) dimension
        vars.push(["d",j,"s",i,"=(t",j,"p",idx,"-s",pidx,"*t",j,"p",pidx,")"].join(""))
      }
    }
  }
  code.push("var " + vars.join(","))
  //Scan loop
  for(i=dimension-1; i>=0; --i) { // Start at largest stride and work your way inwards
    idx = order[i]
    code.push(["for(i",i,"=0;i",i,"<s",idx,";++i",i,"){"].join(""))
  }
  //Push body of inner loop
  code.push(body)
  //Advance scan pointers
  for(i=0; i<dimension; ++i) {
    pidx = idx
    idx = order[i]
    for(j=0; j<nargs; ++j) {
      code.push(["p",j,"+=d",j,"s",i].join(""))
    }
    if(has_index) {
      if(i > 0) {
        code.push(["index[",pidx,"]-=s",pidx].join(""))
      }
      code.push(["++index[",idx,"]"].join(""))
    }
    code.push("}")
  }
  return code.join("\n")
}

// Generate "outer" loops that loop over blocks of data, applying "inner" loops to the blocks by manipulating the local variables in such a way that the inner loop only "sees" the current block.
// TODO: If this is used, then the previous declaration (done by generateCwiseOp) of s* is essentially unnecessary.
//       I believe the s* are not used elsewhere (in particular, I don't think they're used in the pre/post parts and "shape" is defined independently), so it would be possible to make defining the s* dependent on what loop method is being used.
function outerFill(matched, order, proc, body) {
  var dimension = order.length
    , nargs = proc.arrayArgs.length
    , blockSize = proc.blockSize
    , has_index = proc.indexArgs.length > 0
    , code = []
  for(var i=0; i<nargs; ++i) {
    code.push(["var offset",i,"=p",i].join(""))
  }
  //Generate loops for unmatched dimensions
  // The order in which these dimensions are traversed is fairly arbitrary (from small stride to large stride, for the first argument)
  // TODO: It would be nice if the order in which these loops are placed would also be somehow "optimal" (at the very least we should check that it really doesn't hurt us if they're not).
  for(var i=matched; i<dimension; ++i) {
    code.push(["for(var j"+i+"=SS[", order[i], "]|0;j", i, ">0;){"].join("")) // Iterate back to front
    code.push(["if(j",i,"<",blockSize,"){"].join("")) // Either decrease j by blockSize (s = blockSize), or set it to zero (after setting s = j).
    code.push(["s",order[i],"=j",i].join(""))
    code.push(["j",i,"=0"].join(""))
    code.push(["}else{s",order[i],"=",blockSize].join(""))
    code.push(["j",i,"-=",blockSize,"}"].join(""))
    if(has_index) {
      code.push(["index[",order[i],"]=j",i].join(""))
    }
  }
  for(var i=0; i<nargs; ++i) {
    var indexStr = ["offset"+i]
    for(var j=matched; j<dimension; ++j) {
      indexStr.push(["j",j,"*t",i,"p",order[j]].join(""))
    }
    code.push(["p",i,"=(",indexStr.join("+"),")"].join(""))
  }
  code.push(innerFill(order, proc, body))
  for(var i=matched; i<dimension; ++i) {
    code.push("}")
  }
  return code.join("\n")
}

//Count the number of compatible inner orders
// This is the length of the longest common prefix of the arrays in orders.
// Each array in orders lists the dimensions of the correspond ndarray in order of increasing stride.
// This is thus the maximum number of dimensions that can be efficiently traversed by simple nested loops for all arrays.
function countMatches(orders) {
  var matched = 0, dimension = orders[0].length
  while(matched < dimension) {
    for(var j=1; j<orders.length; ++j) {
      if(orders[j][matched] !== orders[0][matched]) {
        return matched
      }
    }
    ++matched
  }
  return matched
}

//Processes a block according to the given data types
// Replaces variable names by different ones, either "local" ones (that are then ferried in and out of the given array) or ones matching the arguments that the function performing the ultimate loop will accept.
function processBlock(block, proc, dtypes) {
  var code = block.body
  var pre = []
  var post = []
  for(var i=0; i<block.args.length; ++i) {
    var carg = block.args[i]
    if(carg.count <= 0) {
      continue
    }
    var re = new RegExp(carg.name, "g")
    var ptrStr = ""
    var arrNum = proc.arrayArgs.indexOf(i)
    switch(proc.argTypes[i]) {
      case "offset":
        var offArgIndex = proc.offsetArgIndex.indexOf(i)
        var offArg = proc.offsetArgs[offArgIndex]
        arrNum = offArg.array
        ptrStr = "+q" + offArgIndex // Adds offset to the "pointer" in the array
      case "array":
        ptrStr = "p" + arrNum + ptrStr
        var localStr = "l" + i
        var arrStr = "a" + arrNum
        if (proc.arrayBlockIndices[arrNum] === 0) { // Argument to body is just a single value from this array
          if(carg.count === 1) { // Argument/array used only once(?)
            if(dtypes[arrNum] === "generic") {
              if(carg.lvalue) {
                pre.push(["var ", localStr, "=", arrStr, ".get(", ptrStr, ")"].join("")) // Is this necessary if the argument is ONLY used as an lvalue? (keep in mind that we can have a += something, so we would actually need to check carg.rvalue)
                code = code.replace(re, localStr)
                post.push([arrStr, ".set(", ptrStr, ",", localStr,")"].join(""))
              } else {
                code = code.replace(re, [arrStr, ".get(", ptrStr, ")"].join(""))
              }
            } else {
              code = code.replace(re, [arrStr, "[", ptrStr, "]"].join(""))
            }
          } else if(dtypes[arrNum] === "generic") {
            pre.push(["var ", localStr, "=", arrStr, ".get(", ptrStr, ")"].join("")) // TODO: Could we optimize by checking for carg.rvalue?
            code = code.replace(re, localStr)
            if(carg.lvalue) {
              post.push([arrStr, ".set(", ptrStr, ",", localStr,")"].join(""))
            }
          } else {
            pre.push(["var ", localStr, "=", arrStr, "[", ptrStr, "]"].join("")) // TODO: Could we optimize by checking for carg.rvalue?
            code = code.replace(re, localStr)
            if(carg.lvalue) {
              post.push([arrStr, "[", ptrStr, "]=", localStr].join(""))
            }
          }
        } else { // Argument to body is a "block"
          var reStrArr = [carg.name], ptrStrArr = [ptrStr]
          for(var j=0; j<Math.abs(proc.arrayBlockIndices[arrNum]); j++) {
            reStrArr.push("\\s*\\[([^\\]]+)\\]")
            ptrStrArr.push("$" + (j+1) + "*t" + arrNum + "b" + j) // Matched index times stride
          }
          re = new RegExp(reStrArr.join(""), "g")
          ptrStr = ptrStrArr.join("+")
          if(dtypes[arrNum] === "generic") {
            /*if(carg.lvalue) {
              pre.push(["var ", localStr, "=", arrStr, ".get(", ptrStr, ")"].join("")) // Is this necessary if the argument is ONLY used as an lvalue? (keep in mind that we can have a += something, so we would actually need to check carg.rvalue)
              code = code.replace(re, localStr)
              post.push([arrStr, ".set(", ptrStr, ",", localStr,")"].join(""))
            } else {
              code = code.replace(re, [arrStr, ".get(", ptrStr, ")"].join(""))
            }*/
            throw new Error("cwise: Generic arrays not supported in combination with blocks!")
          } else {
            // This does not produce any local variables, even if variables are used multiple times. It would be possible to do so, but it would complicate things quite a bit.
            code = code.replace(re, [arrStr, "[", ptrStr, "]"].join(""))
          }
        }
      break
      case "scalar":
        code = code.replace(re, "Y" + proc.scalarArgs.indexOf(i))
      break
      case "index":
        code = code.replace(re, "index")
      break
      case "shape":
        code = code.replace(re, "shape")
      break
    }
  }
  return [pre.join("\n"), code, post.join("\n")].join("\n").trim()
}

function typeSummary(dtypes) {
  var summary = new Array(dtypes.length)
  var allEqual = true
  for(var i=0; i<dtypes.length; ++i) {
    var t = dtypes[i]
    var digits = t.match(/\d+/)
    if(!digits) {
      digits = ""
    } else {
      digits = digits[0]
    }
    if(t.charAt(0) === 0) {
      summary[i] = "u" + t.charAt(1) + digits
    } else {
      summary[i] = t.charAt(0) + digits
    }
    if(i > 0) {
      allEqual = allEqual && summary[i] === summary[i-1]
    }
  }
  if(allEqual) {
    return summary[0]
  }
  return summary.join("")
}

//Generates a cwise operator
function generateCWiseOp(proc, typesig) {

  //Compute dimension
  // Arrays get put first in typesig, and there are two entries per array (dtype and order), so this gets the number of dimensions in the first array arg.
  var dimension = (typesig[1].length - Math.abs(proc.arrayBlockIndices[0]))|0
  var orders = new Array(proc.arrayArgs.length)
  var dtypes = new Array(proc.arrayArgs.length)
  for(var i=0; i<proc.arrayArgs.length; ++i) {
    dtypes[i] = typesig[2*i]
    orders[i] = typesig[2*i+1]
  }
  
  //Determine where block and loop indices start and end
  var blockBegin = [], blockEnd = [] // These indices are exposed as blocks
  var loopBegin = [], loopEnd = [] // These indices are iterated over
  var loopOrders = [] // orders restricted to the loop indices
  for(var i=0; i<proc.arrayArgs.length; ++i) {
    if (proc.arrayBlockIndices[i]<0) {
      loopBegin.push(0)
      loopEnd.push(dimension)
      blockBegin.push(dimension)
      blockEnd.push(dimension+proc.arrayBlockIndices[i])
    } else {
      loopBegin.push(proc.arrayBlockIndices[i]) // Non-negative
      loopEnd.push(proc.arrayBlockIndices[i]+dimension)
      blockBegin.push(0)
      blockEnd.push(proc.arrayBlockIndices[i])
    }
    var newOrder = []
    for(var j=0; j<orders[i].length; j++) {
      if (loopBegin[i]<=orders[i][j] && orders[i][j]<loopEnd[i]) {
        newOrder.push(orders[i][j]-loopBegin[i]) // If this is a loop index, put it in newOrder, subtracting loopBegin, to make sure that all loopOrders are using a common set of indices.
      }
    }
    loopOrders.push(newOrder)
  }

  //First create arguments for procedure
  var arglist = ["SS"] // SS is the overall shape over which we iterate
  var code = ["'use strict'"]
  var vars = []
  
  for(var j=0; j<dimension; ++j) {
    vars.push(["s", j, "=SS[", j, "]"].join("")) // The limits for each dimension.
  }
  for(var i=0; i<proc.arrayArgs.length; ++i) {
    arglist.push("a"+i) // Actual data array
    arglist.push("t"+i) // Strides
    arglist.push("p"+i) // Offset in the array at which the data starts (also used for iterating over the data)
    
    for(var j=0; j<dimension; ++j) { // Unpack the strides into vars for looping
      vars.push(["t",i,"p",j,"=t",i,"[",loopBegin[i]+j,"]"].join(""))
    }
    
    for(var j=0; j<Math.abs(proc.arrayBlockIndices[i]); ++j) { // Unpack the strides into vars for block iteration
      vars.push(["t",i,"b",j,"=t",i,"[",blockBegin[i]+j,"]"].join(""))
    }
  }
  for(var i=0; i<proc.scalarArgs.length; ++i) {
    arglist.push("Y" + i)
  }
  if(proc.shapeArgs.length > 0) {
    vars.push("shape=SS.slice(0)") // Makes the shape over which we iterate available to the user defined functions (so you can use width/height for example)
  }
  if(proc.indexArgs.length > 0) {
    // Prepare an array to keep track of the (logical) indices, initialized to dimension zeroes.
    var zeros = new Array(dimension)
    for(var i=0; i<dimension; ++i) {
      zeros[i] = "0"
    }
    vars.push(["index=[", zeros.join(","), "]"].join(""))
  }
  for(var i=0; i<proc.offsetArgs.length; ++i) { // Offset arguments used for stencil operations
    var off_arg = proc.offsetArgs[i]
    var init_string = []
    for(var j=0; j<off_arg.offset.length; ++j) {
      if(off_arg.offset[j] === 0) {
        continue
      } else if(off_arg.offset[j] === 1) {
        init_string.push(["t", off_arg.array, "p", j].join(""))      
      } else {
        init_string.push([off_arg.offset[j], "*t", off_arg.array, "p", j].join(""))
      }
    }
    if(init_string.length === 0) {
      vars.push("q" + i + "=0")
    } else {
      vars.push(["q", i, "=", init_string.join("+")].join(""))
    }
  }

  //Prepare this variables
  var thisVars = uniq([].concat(proc.pre.thisVars)
                      .concat(proc.body.thisVars)
                      .concat(proc.post.thisVars))
  vars = vars.concat(thisVars)
  code.push("var " + vars.join(","))
  for(var i=0; i<proc.arrayArgs.length; ++i) {
    code.push("p"+i+"|=0")
  }
  
  //Inline prelude
  if(proc.pre.body.length > 3) {
    code.push(processBlock(proc.pre, proc, dtypes))
  }

  //Process body
  var body = processBlock(proc.body, proc, dtypes)
  var matched = countMatches(loopOrders)
  if(matched < dimension) {
    code.push(outerFill(matched, loopOrders[0], proc, body)) // TODO: Rather than passing loopOrders[0], it might be interesting to look at passing an order that represents the majority of the arguments for example.
  } else {
    code.push(innerFill(loopOrders[0], proc, body))
  }

  //Inline epilog
  if(proc.post.body.length > 3) {
    code.push(processBlock(proc.post, proc, dtypes))
  }
  
  if(proc.debug) {
    console.log("-----Generated cwise routine for ", typesig, ":\n" + code.join("\n") + "\n----------")
  }
  
  var loopName = [(proc.funcName||"unnamed"), "_cwise_loop_", orders[0].join("s"),"m",matched,typeSummary(dtypes)].join("")
  var f = new Function(["function ",loopName,"(", arglist.join(","),"){", code.join("\n"),"} return ", loopName].join(""))
  return f()
}
module.exports = generateCWiseOp

},{"uniq":22}],7:[function(require,module,exports){
"use strict"

// The function below is called when constructing a cwise function object, and does the following:
// A function object is constructed which accepts as argument a compilation function and returns another function.
// It is this other function that is eventually returned by createThunk, and this function is the one that actually
// checks whether a certain pattern of arguments has already been used before and compiles new loops as needed.
// The compilation passed to the first function object is used for compiling new functions.
// Once this function object is created, it is called with compile as argument, where the first argument of compile
// is bound to "proc" (essentially containing a preprocessed version of the user arguments to cwise).
// So createThunk roughly works like this:
// function createThunk(proc) {
//   var thunk = function(compileBound) {
//     var CACHED = {}
//     return function(arrays and scalars) {
//       if (dtype and order of arrays in CACHED) {
//         var func = CACHED[dtype and order of arrays]
//       } else {
//         var func = CACHED[dtype and order of arrays] = compileBound(dtype and order of arrays)
//       }
//       return func(arrays and scalars)
//     }
//   }
//   return thunk(compile.bind1(proc))
// }

var compile = require("./compile.js")

function createThunk(proc) {
  var code = ["'use strict'", "var CACHED={}"]
  var vars = []
  var thunkName = proc.funcName + "_cwise_thunk"
  
  //Build thunk
  code.push(["return function ", thunkName, "(", proc.shimArgs.join(","), "){"].join(""))
  var typesig = []
  var string_typesig = []
  var proc_args = [["array",proc.arrayArgs[0],".shape.slice(", // Slice shape so that we only retain the shape over which we iterate (which gets passed to the cwise operator as SS).
                    Math.max(0,proc.arrayBlockIndices[0]),proc.arrayBlockIndices[0]<0?(","+proc.arrayBlockIndices[0]+")"):")"].join("")]
  var shapeLengthConditions = [], shapeConditions = []
  // Process array arguments
  for(var i=0; i<proc.arrayArgs.length; ++i) {
    var j = proc.arrayArgs[i]
    vars.push(["t", j, "=array", j, ".dtype,",
               "r", j, "=array", j, ".order"].join(""))
    typesig.push("t" + j)
    typesig.push("r" + j)
    string_typesig.push("t"+j)
    string_typesig.push("r"+j+".join()")
    proc_args.push("array" + j + ".data")
    proc_args.push("array" + j + ".stride")
    proc_args.push("array" + j + ".offset|0")
    if (i>0) { // Gather conditions to check for shape equality (ignoring block indices)
      shapeLengthConditions.push("array" + proc.arrayArgs[0] + ".shape.length===array" + j + ".shape.length+" + (Math.abs(proc.arrayBlockIndices[0])-Math.abs(proc.arrayBlockIndices[i])))
      shapeConditions.push("array" + proc.arrayArgs[0] + ".shape[shapeIndex+" + Math.max(0,proc.arrayBlockIndices[0]) + "]===array" + j + ".shape[shapeIndex+" + Math.max(0,proc.arrayBlockIndices[i]) + "]")
    }
  }
  // Check for shape equality
  if (proc.arrayArgs.length > 1) {
    code.push("if (!(" + shapeLengthConditions.join(" && ") + ")) throw new Error('cwise: Arrays do not all have the same dimensionality!')")
    code.push("for(var shapeIndex=array" + proc.arrayArgs[0] + ".shape.length-" + Math.abs(proc.arrayBlockIndices[0]) + "; shapeIndex-->0;) {")
    code.push("if (!(" + shapeConditions.join(" && ") + ")) throw new Error('cwise: Arrays do not all have the same shape!')")
    code.push("}")
  }
  // Process scalar arguments
  for(var i=0; i<proc.scalarArgs.length; ++i) {
    proc_args.push("scalar" + proc.scalarArgs[i])
  }
  // Check for cached function (and if not present, generate it)
  vars.push(["type=[", string_typesig.join(","), "].join()"].join(""))
  vars.push("proc=CACHED[type]")
  code.push("var " + vars.join(","))
  
  code.push(["if(!proc){",
             "CACHED[type]=proc=compile([", typesig.join(","), "])}",
             "return proc(", proc_args.join(","), ")}"].join(""))

  if(proc.debug) {
    console.log("-----Generated thunk:\n" + code.join("\n") + "\n----------")
  }
  
  //Compile thunk
  var thunk = new Function("compile", code.join("\n"))
  return thunk(compile.bind(undefined, proc))
}

module.exports = createThunk

},{"./compile.js":6}],8:[function(require,module,exports){
module.exports = require("cwise-compiler")
},{"cwise-compiler":5}],9:[function(require,module,exports){
"use strict"

function dupe_array(count, value, i) {
  var c = count[i]|0
  if(c <= 0) {
    return []
  }
  var result = new Array(c), j
  if(i === count.length-1) {
    for(j=0; j<c; ++j) {
      result[j] = value
    }
  } else {
    for(j=0; j<c; ++j) {
      result[j] = dupe_array(count, value, i+1)
    }
  }
  return result
}

function dupe_number(count, value) {
  var result, i
  result = new Array(count)
  for(i=0; i<count; ++i) {
    result[i] = value
  }
  return result
}

function dupe(count, value) {
  if(typeof value === "undefined") {
    value = 0
  }
  switch(typeof count) {
    case "number":
      if(count > 0) {
        return dupe_number(count|0, value)
      }
    break
    case "object":
      if(typeof (count.length) === "number") {
        return dupe_array(count, value, 0)
      }
    break
  }
  return []
}

module.exports = dupe
},{}],10:[function(require,module,exports){
exports.read = function (buffer, offset, isLE, mLen, nBytes) {
  var e, m
  var eLen = nBytes * 8 - mLen - 1
  var eMax = (1 << eLen) - 1
  var eBias = eMax >> 1
  var nBits = -7
  var i = isLE ? (nBytes - 1) : 0
  var d = isLE ? -1 : 1
  var s = buffer[offset + i]

  i += d

  e = s & ((1 << (-nBits)) - 1)
  s >>= (-nBits)
  nBits += eLen
  for (; nBits > 0; e = e * 256 + buffer[offset + i], i += d, nBits -= 8) {}

  m = e & ((1 << (-nBits)) - 1)
  e >>= (-nBits)
  nBits += mLen
  for (; nBits > 0; m = m * 256 + buffer[offset + i], i += d, nBits -= 8) {}

  if (e === 0) {
    e = 1 - eBias
  } else if (e === eMax) {
    return m ? NaN : ((s ? -1 : 1) * Infinity)
  } else {
    m = m + Math.pow(2, mLen)
    e = e - eBias
  }
  return (s ? -1 : 1) * m * Math.pow(2, e - mLen)
}

exports.write = function (buffer, value, offset, isLE, mLen, nBytes) {
  var e, m, c
  var eLen = nBytes * 8 - mLen - 1
  var eMax = (1 << eLen) - 1
  var eBias = eMax >> 1
  var rt = (mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0)
  var i = isLE ? 0 : (nBytes - 1)
  var d = isLE ? 1 : -1
  var s = value < 0 || (value === 0 && 1 / value < 0) ? 1 : 0

  value = Math.abs(value)

  if (isNaN(value) || value === Infinity) {
    m = isNaN(value) ? 1 : 0
    e = eMax
  } else {
    e = Math.floor(Math.log(value) / Math.LN2)
    if (value * (c = Math.pow(2, -e)) < 1) {
      e--
      c *= 2
    }
    if (e + eBias >= 1) {
      value += rt / c
    } else {
      value += rt * Math.pow(2, 1 - eBias)
    }
    if (value * c >= 2) {
      e++
      c /= 2
    }

    if (e + eBias >= eMax) {
      m = 0
      e = eMax
    } else if (e + eBias >= 1) {
      m = (value * c - 1) * Math.pow(2, mLen)
      e = e + eBias
    } else {
      m = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen)
      e = 0
    }
  }

  for (; mLen >= 8; buffer[offset + i] = m & 0xff, i += d, m /= 256, mLen -= 8) {}

  e = (e << mLen) | m
  eLen += mLen
  for (; eLen > 0; buffer[offset + i] = e & 0xff, i += d, e /= 256, eLen -= 8) {}

  buffer[offset + i - d] |= s * 128
}

},{}],11:[function(require,module,exports){
"use strict"

function iota(n) {
  var result = new Array(n)
  for(var i=0; i<n; ++i) {
    result[i] = i
  }
  return result
}

module.exports = iota
},{}],12:[function(require,module,exports){
/*!
 * Determine if an object is a Buffer
 *
 * @author   Feross Aboukhadijeh <feross@feross.org> <http://feross.org>
 * @license  MIT
 */

// The _isBuffer check is for Safari 5-7 support, because it's missing
// Object.prototype.constructor. Remove this eventually
module.exports = function (obj) {
  return obj != null && (isBuffer(obj) || isSlowBuffer(obj) || !!obj._isBuffer)
}

function isBuffer (obj) {
  return !!obj.constructor && typeof obj.constructor.isBuffer === 'function' && obj.constructor.isBuffer(obj)
}

// For Node v0.10 support. Remove this eventually.
function isSlowBuffer (obj) {
  return typeof obj.readFloatLE === 'function' && typeof obj.slice === 'function' && isBuffer(obj.slice(0, 0))
}

},{}],13:[function(require,module,exports){
'use strict'

var ops = require('ndarray-ops')
var ndarray = require('ndarray')
var pool = require('typedarray-pool')
var fftm = require('./lib/fft-matrix.js')

function ndfft(dir, x, y) {
  var shape = x.shape
    , d = shape.length
    , size = 1
    , stride = new Array(d)
    , pad = 0
    , i, j
  for(i=d-1; i>=0; --i) {
    stride[i] = size
    size *= shape[i]
    pad = Math.max(pad, fftm.scratchMemory(shape[i]))
    if(x.shape[i] !== y.shape[i]) {
      throw new Error('Shape mismatch, real and imaginary arrays must have same size')
    }
  }
  var buf_size = 4 * size + pad
  var buffer
  if( x.dtype === 'array' ||
      x.dtype === 'float64' ||
      x.dtype === 'custom' ) {
    buffer = pool.mallocDouble(buf_size)
  } else {
    buffer = pool.mallocFloat(buf_size)
  }
  var x1 = ndarray(buffer, shape.slice(0), stride, 0)
    , y1 = ndarray(buffer, shape.slice(0), stride.slice(0), size)
    , x2 = ndarray(buffer, shape.slice(0), stride.slice(0), 2*size)
    , y2 = ndarray(buffer, shape.slice(0), stride.slice(0), 3*size)
    , tmp, n, s1, s2
    , scratch_ptr = 4 * size
  
  //Copy into x1/y1
  ops.assign(x1, x)
  ops.assign(y1, y)
  
  for(i=d-1; i>=0; --i) {
    fftm(dir, size/shape[i], shape[i], buffer, x1.offset, y1.offset, scratch_ptr)
    if(i === 0) {
      break
    }
    
    //Compute new stride for x2/y2
    n = 1
    s1 = x2.stride
    s2 = y2.stride
    for(j=i-1; j<d; ++j) {
      s2[j] = s1[j] = n
      n *= shape[j]
    }
    for(j=i-2; j>=0; --j) {
      s2[j] = s1[j] = n
      n *= shape[j]
    }
    
    //Transpose
    ops.assign(x2, x1)
    ops.assign(y2, y1)
    
    //Swap buffers
    tmp = x1
    x1 = x2
    x2 = tmp
    tmp = y1
    y1 = y2
    y2 = tmp
  }
  
  //Copy result back into x
  ops.assign(x, x1)
  ops.assign(y, y1)
  
  pool.free(buffer)
}

module.exports = ndfft
},{"./lib/fft-matrix.js":14,"ndarray":18,"ndarray-ops":17,"typedarray-pool":21}],14:[function(require,module,exports){
var bits = require('bit-twiddle')

function fft(dir, nrows, ncols, buffer, x_ptr, y_ptr, scratch_ptr) {
  dir |= 0
  nrows |= 0
  ncols |= 0
  x_ptr |= 0
  y_ptr |= 0
  if(bits.isPow2(ncols)) {
    fftRadix2(dir, nrows, ncols, buffer, x_ptr, y_ptr)
  } else {
    fftBluestein(dir, nrows, ncols, buffer, x_ptr, y_ptr, scratch_ptr)
  }
}
module.exports = fft

function scratchMemory(n) {
  if(bits.isPow2(n)) {
    return 0
  }
  return 2 * n + 4 * bits.nextPow2(2*n + 1)
}
module.exports.scratchMemory = scratchMemory


//Radix 2 FFT Adapted from Paul Bourke's C Implementation
function fftRadix2(dir, nrows, ncols, buffer, x_ptr, y_ptr) {
  dir |= 0
  nrows |= 0
  ncols |= 0
  x_ptr |= 0
  y_ptr |= 0
  var nn,m,i,i1,j,k,i2,l,l1,l2
  var c1,c2,t,t1,t2,u1,u2,z,row,a,b,c,d,k1,k2,k3
  
  // Calculate the number of points
  nn = ncols
  m = bits.log2(nn)
  
  for(row=0; row<nrows; ++row) {  
    // Do the bit reversal
    i2 = nn >> 1;
    j = 0;
    for(i=0;i<nn-1;i++) {
      if(i < j) {
        t = buffer[x_ptr+i]
        buffer[x_ptr+i] = buffer[x_ptr+j]
        buffer[x_ptr+j] = t
        t = buffer[y_ptr+i]
        buffer[y_ptr+i] = buffer[y_ptr+j]
        buffer[y_ptr+j] = t
      }
      k = i2
      while(k <= j) {
        j -= k
        k >>= 1
      }
      j += k
    }
    
    // Compute the FFT
    c1 = -1.0
    c2 = 0.0
    l2 = 1
    for(l=0;l<m;l++) {
      l1 = l2
      l2 <<= 1
      u1 = 1.0
      u2 = 0.0
      for(j=0;j<l1;j++) {
        for(i=j;i<nn;i+=l2) {
          i1 = i + l1
          a = buffer[x_ptr+i1]
          b = buffer[y_ptr+i1]
          c = buffer[x_ptr+i]
          d = buffer[y_ptr+i]
          k1 = u1 * (a + b)
          k2 = a * (u2 - u1)
          k3 = b * (u1 + u2)
          t1 = k1 - k3
          t2 = k1 + k2
          buffer[x_ptr+i1] = c - t1
          buffer[y_ptr+i1] = d - t2
          buffer[x_ptr+i] += t1
          buffer[y_ptr+i] += t2
        }
        k1 = c1 * (u1 + u2)
        k2 = u1 * (c2 - c1)
        k3 = u2 * (c1 + c2)
        u1 = k1 - k3
        u2 = k1 + k2
      }
      c2 = Math.sqrt((1.0 - c1) / 2.0)
      if(dir < 0) {
        c2 = -c2
      }
      c1 = Math.sqrt((1.0 + c1) / 2.0)
    }
    
    // Scaling for inverse transform
    if(dir < 0) {
      var scale_f = 1.0 / nn
      for(i=0;i<nn;i++) {
        buffer[x_ptr+i] *= scale_f
        buffer[y_ptr+i] *= scale_f
      }
    }
    
    // Advance pointers
    x_ptr += ncols
    y_ptr += ncols
  }
}

// Use Bluestein algorithm for npot FFTs
// Scratch memory required:  2 * ncols + 4 * bits.nextPow2(2*ncols + 1)
function fftBluestein(dir, nrows, ncols, buffer, x_ptr, y_ptr, scratch_ptr) {
  dir |= 0
  nrows |= 0
  ncols |= 0
  x_ptr |= 0
  y_ptr |= 0
  scratch_ptr |= 0

  // Initialize tables
  var m = bits.nextPow2(2 * ncols + 1)
    , cos_ptr = scratch_ptr
    , sin_ptr = cos_ptr + ncols
    , xs_ptr  = sin_ptr + ncols
    , ys_ptr  = xs_ptr  + m
    , cft_ptr = ys_ptr  + m
    , sft_ptr = cft_ptr + m
    , w = -dir * Math.PI / ncols
    , row, a, b, c, d, k1, k2, k3
    , i
  for(i=0; i<ncols; ++i) {
    a = w * ((i * i) % (ncols * 2))
    c = Math.cos(a)
    d = Math.sin(a)
    buffer[cft_ptr+(m-i)] = buffer[cft_ptr+i] = buffer[cos_ptr+i] = c
    buffer[sft_ptr+(m-i)] = buffer[sft_ptr+i] = buffer[sin_ptr+i] = d
  }
  for(i=ncols; i<=m-ncols; ++i) {
    buffer[cft_ptr+i] = 0.0
  }
  for(i=ncols; i<=m-ncols; ++i) {
    buffer[sft_ptr+i] = 0.0
  }

  fftRadix2(1, 1, m, buffer, cft_ptr, sft_ptr)
  
  //Compute scale factor
  if(dir < 0) {
    w = 1.0 / ncols
  } else {
    w = 1.0
  }
  
  //Handle direction
  for(row=0; row<nrows; ++row) {
  
    // Copy row into scratch memory, multiply weights
    for(i=0; i<ncols; ++i) {
      a = buffer[x_ptr+i]
      b = buffer[y_ptr+i]
      c = buffer[cos_ptr+i]
      d = -buffer[sin_ptr+i]
      k1 = c * (a + b)
      k2 = a * (d - c)
      k3 = b * (c + d)
      buffer[xs_ptr+i] = k1 - k3
      buffer[ys_ptr+i] = k1 + k2
    }
    //Zero out the rest
    for(i=ncols; i<m; ++i) {
      buffer[xs_ptr+i] = 0.0
    }
    for(i=ncols; i<m; ++i) {
      buffer[ys_ptr+i] = 0.0
    }
    
    // FFT buffer
    fftRadix2(1, 1, m, buffer, xs_ptr, ys_ptr)
    
    // Apply multiplier
    for(i=0; i<m; ++i) {
      a = buffer[xs_ptr+i]
      b = buffer[ys_ptr+i]
      c = buffer[cft_ptr+i]
      d = buffer[sft_ptr+i]
      k1 = c * (a + b)
      k2 = a * (d - c)
      k3 = b * (c + d)
      buffer[xs_ptr+i] = k1 - k3
      buffer[ys_ptr+i] = k1 + k2
    }
    
    // Inverse FFT buffer
    fftRadix2(-1, 1, m, buffer, xs_ptr, ys_ptr)
    
    // Copy result back into x/y
    for(i=0; i<ncols; ++i) {
      a = buffer[xs_ptr+i]
      b = buffer[ys_ptr+i]
      c = buffer[cos_ptr+i]
      d = -buffer[sin_ptr+i]
      k1 = c * (a + b)
      k2 = a * (d - c)
      k3 = b * (c + d)
      buffer[x_ptr+i] = w * (k1 - k3)
      buffer[y_ptr+i] = w * (k1 + k2)
    }
    
    x_ptr += ncols
    y_ptr += ncols
  }
}

},{"bit-twiddle":2}],15:[function(require,module,exports){
"use strict"

module.exports = matrixProduct

var generatePlan = require("./lib/planner.js")

function shape(arr) {
  if(Array.isArray(arr)) {
    return [ arr.length, arr[0].length ]
  } else {
    return arr.shape
  }
}

function checkShapes(out, a, b) {
  var os = shape(out)
  var as = shape(a)
  var bs = shape(b)
  if(os[0] !== as[0] || os[1] !== bs[1] || as[1] !== bs[0]) {
    throw new Error("Mismatched array shapes for matrix product")
  }
}

function classifyType(m) {
  if(Array.isArray(m)) {
    if(Array.isArray(m)) {
      return [ "r", "native" ]
    }
  } else if(m.shape && (m.shape.length === 2)) {
    if(m.order[0]) {
      return [ "r", m.dtype ]
    } else {
      return [ "c", m.dtype ]
    }
  }
  throw new Error("Unrecognized data type")
}

var CACHE = {}

function matrixProduct(out, a, b, alpha, beta) {
  if(alpha === undefined) {
    alpha = 1.0
  }
  if(beta === undefined) {
    beta = 0.0
  }
  var useAlpha = (alpha !== 1.0)
  var useBeta  = (beta !== 0.0)
  var outType  = classifyType(out)
  var aType    = classifyType(a)
  var bType    = classifyType(b)

  checkShapes(out, a, b)

  var typeSig  = [ outType, aType, bType, useAlpha, useBeta ].join(":")
  var proc     = CACHE[typeSig]
  if(!proc) {
    proc = CACHE[typeSig] = generatePlan(outType, aType, bType, useAlpha, useBeta)
  }
  return proc(out, a, b, alpha, beta)
}
},{"./lib/planner.js":16}],16:[function(require,module,exports){
"use strict"

module.exports = generateMatrixProduct

var BLOCK_SIZE = 32

function unpackOrder(order) {
  return order === "r" ? [1,0] : [0,1]
}

function unpackShape(name, type) {
  if(type[1] === "native") {
    return [
      name, "d0=", name, ".length,",
      name, "d1=", name, "[0].length,"
    ].join("")
  } else {
    return [
      name, "d0=", name, ".shape[0],",
      name, "d1=", name, ".shape[1],",
      name, "s0=", name, ".stride[0],",
      name, "s1=", name, ".stride[1],",
      name, "o=", name, ".offset,",
      name, "d=", name, ".data,"
    ].join("")
  }
}

function start(order, name, type, i, j, w) {
  var code = []
  if(type[1] === "native") {
    if(order[0]) {
      if(i) {
        code.push("var ", name, "p=", name, "[", i, "];")
      } else {
        code.push("var ", name, "p=", name, "[0];")
      }
    }
  } else {
    if(i && j) {
      if(w) {
        code.push(
          "var ", name, "t0=", name, "s", order[0], ",",
                  name, "t1=", name, "s", order[1], "-", name, "s", order[0], "*", w, ",",
                  name, "p=", name, "o+", i, "*", name, "s0+", j, "*", name, "s1;")
      } else {
        code.push(
          "var ", name, "t0=", name, "s", order[0], ",",
                  name, "p=", name, "o+", i, "*", name, "s0+", j, "*", name, "s1;")
      }
    } else if(i) {
      code.push(
        "var ", name, "t0=", name, "s", order[0], ",",
                name, "p=", name, "o+", i, "*", name, "s0;")
    } else if(j) {
      code.push(
        "var ", name, "t0=", name, "s", order[0], ",",
                name, "p=", name, "o+", j, "*", name, "s1;")
    } else  {
      code.push(
        "var ", name, "t0=", name, "s", order[0], ",",
                name, "t1=", name, "s", order[1], "-", name, "s", order[0], "*", name, "d", order[0], ",",
                name, "p=", name, "o;")
    }
  }
  return code
}

function walk(order, name, type, d, i) {
  var code = []
  if(type[1] === "native") {
    if(order[0] && d === 1) {
      code.push(name, "p=", name, "[", i, "+1]")
    }
  } else {
    code.push(name, "p+=", name, "t", d, ";")
  }
  return code
}

function write(order, name, type, i, j, w) {
  var code = []
  if(type[1] === "native") {
    if(order[0]) {
      code.push(name, "p[", j, "]=", w, ";")
    } else {
      code.push(name, "[", i, "][", j, "]=", w, ";")
    }
  } else if(type[1] === "generic") {
    code.push(name, "d.set(", name, "p,", w, ");")
  } else {
    code.push(name, "d[", name, "p]=", w, ";")
  }
  return code
}

function read(order, name, type, i, j) {
  var code = []
  if(type[1] === "native") {
    if(order[0]) {
      code.push(name, "p[", j, "]")
    } else {
      code.push(name, "[", i, "][", j, "]")
    }
  } else if(type[1] === "generic") {
    code.push(name, "d.get(", name, "p)")
  } else {
    code.push(name, "d[", name, "p]")
  }
  return code.join("")
}

function generateRowColumnLoop(oType, aType, bType, useAlpha, useBeta) {
  var code = []
  var oOrd = oType[0] === "r" ? [1,0] : [0,1], aOrd = [1, 0], bOrd = [0, 1]
  var symbols = ["i", "j"]

  code.push.apply(code, start(oOrd, "o", oType))
  
  if(oOrd[1]) {
    code.push("for(j=0;j<od1;++j){")
    code.push("for(i=0;i<od0;++i){")
  } else {
    code.push("for(i=0;i<od0;++i){")
    code.push("for(j=0;j<od1;++j){")
  }

  code.push.apply(code, start(aOrd, "a", aType, "i"))
  code.push.apply(code, start(bOrd, "b", bType, undefined, "j"))

  code.push(
      "var r=0.0;",
      "for(k=0;k<ad1;++k){",
      "r+=", 
        read(aOrd, "a", aType, "i", "k"), "*", 
        read(bOrd, "b", bType, "k", "j"), ";")

  //Terminate k loop
  code.push.apply(code, walk(aOrd, "a", aType, 0, "k"))
  code.push.apply(code, walk(bOrd, "b", bType, 0, "k"))
  code.push("}")

  //Write r to output
  if(useAlpha) {
    code.push("r*=A;")
  }
  if(useBeta) {
    code.push("r+=B*", read(oOrd, "o", oType, "i", "j"), ";")
  }
  code.push.apply(code, write(oOrd, "o", oType, "i", "j", "r"))
  
  //Terminate j loop loop
  code.push.apply(code, walk(oOrd, "o", oType, 0, symbols[1]))
  code.push("}")

  //Terminate i loop
  code.push.apply(code, walk(oOrd, "o", oType, 1, symbols[0]))
  code.push("}")

  return code
}

function generateBetaPass(oType, useBeta) {
  var code = []
  var oOrd = oType[0] === "r" ? [1,0] : [0,1], symbols
  if(useBeta) {
    code.push("if(B!==1.0){")
  }
  code.push.apply(code, start(oOrd, "o", oType))
  if(oOrd[0]) {
    code.push("for(i=0;i<od0;++i){for(j=0;j<od1;++j){")
    symbols = ["i", "j"]
  } else {
    code.push("for(j=0;j<od1;++j){for(i=0;i<od0;++i){")
    symbols = ["j", "i"]
  }
  if(useBeta) {
    code.push.apply(code, write(oOrd, "o", oType, "i", "j", 
      "B*"+read(oOrd, "o", oType, "i", "j")))
  } else {
    code.push.apply(code, write(oOrd, "o", oType, "i", "j", "0"))
  }
  code.push.apply(code, walk(oOrd, "o", oType, 0, symbols[1]))
  code.push("}")
  code.push.apply(code, walk(oOrd, "o", oType, 1, symbols[0]))
  code.push("}")
  if(useBeta) {
    code.push("}")
  }
  return code
}

function generateBlockLoop(oType, aType, bType, useAlpha, useBeta) {
  var code = []
  var shapes = [ "od0", "od1", "ad1" ]
  var oOrd = [1, 0]
  var aOrd = [1, 0]
  var bOrd = [0, 1]

  //Do pass over output to zero it out
  code.push.apply(code, generateBetaPass(oType, useBeta))

  for(var i=0; i<3; ++i) {
    code.push(
      "for(var i", i, "=", shapes[i], ";i", i, ">0;){",
        "var w", i, "=", BLOCK_SIZE, ";",
        "if(i", i, "<", BLOCK_SIZE, "){",
          "w", i, "=i", i, ";",
          "i", i, "=0;",
        "}else{",
          "i", i, "-=", BLOCK_SIZE, ";",
        "}")
  }

  code.push.apply(code, start(oOrd, "o", oType, "i0", "i1", "w1"))
  
  code.push("for(i=0;i<w0;++i){\
for(j=0;j<w1;++j){\
var r=0.0;")

  code.push.apply(code, start(aOrd, "a", aType, "(i0+i)", "i2"))
  code.push.apply(code, start(bOrd, "b", bType, "i2", "(i1+j)"))

  code.push("for(k=0;k<w2;++k){")

  code.push("r+=",
    read(aOrd, "a", aType, "(i0+i)", "(i2+k)"), "*", 
    read(bOrd, "b", bType, "(i2+k)", "(i1+j)"), ";")

  //Close off k-loop
  code.push.apply(code, walk(aOrd, "a", aType, 0, "(i2+k)"))
  code.push.apply(code, walk(bOrd, "b", bType, 0, "(i2+k)"))
  code.push("}")

  //Write r back to output array
  var sym = "r"
  if(useAlpha) {
    sym = "A*r"
  }
  code.push.apply(code, write(oOrd, "o", oType, "(i0+i)", "(i1+j)", 
    sym + "+" + read(oOrd, "o", oType, "(i0+i)", "(i1+j)")))

  //Close off j-loop
  code.push.apply(code, walk(oOrd, "o", oType, 0, "(i1+j)"))
  code.push("}")

  //Close off i-loop
  code.push.apply(code, walk(oOrd, "o", oType, 1, "(i0+i)"))
  code.push("}}}}")

  return code
}

function generateMatrixProduct(outType, aType, bType, useAlpha, useBeta) {
  var funcName = ["gemm", outType[0], outType[1], 
                     "a", aType[0], aType[1],
                     "b", bType[0], bType[1],
                     useAlpha ? "alpha" : "",
                     useBeta ? "beta" : "" ].join("")
  var code = [
    "function ", funcName, "(o,a,b,A,B){",
    "var ", unpackShape("o", outType), 
            unpackShape("a", aType),
            unpackShape("b", bType),
            "i,j,k;"
  ]

  if(aType[0] === "r" && bType[0] === "c") {
    code.push.apply(code, generateRowColumnLoop(outType, aType, bType, useAlpha, useBeta))
  } else {
    code.push.apply(code, generateBlockLoop(outType, aType, bType, useAlpha, useBeta))
  }

  code.push("}return ", funcName)

  //Compile function
  var proc = new Function(code.join(""))
  return proc()
}
},{}],17:[function(require,module,exports){
"use strict"

var compile = require("cwise-compiler")

var EmptyProc = {
  body: "",
  args: [],
  thisVars: [],
  localVars: []
}

function fixup(x) {
  if(!x) {
    return EmptyProc
  }
  for(var i=0; i<x.args.length; ++i) {
    var a = x.args[i]
    if(i === 0) {
      x.args[i] = {name: a, lvalue:true, rvalue: !!x.rvalue, count:x.count||1 }
    } else {
      x.args[i] = {name: a, lvalue:false, rvalue:true, count: 1}
    }
  }
  if(!x.thisVars) {
    x.thisVars = []
  }
  if(!x.localVars) {
    x.localVars = []
  }
  return x
}

function pcompile(user_args) {
  return compile({
    args:     user_args.args,
    pre:      fixup(user_args.pre),
    body:     fixup(user_args.body),
    post:     fixup(user_args.proc),
    funcName: user_args.funcName
  })
}

function makeOp(user_args) {
  var args = []
  for(var i=0; i<user_args.args.length; ++i) {
    args.push("a"+i)
  }
  var wrapper = new Function("P", [
    "return function ", user_args.funcName, "_ndarrayops(", args.join(","), ") {P(", args.join(","), ");return a0}"
  ].join(""))
  return wrapper(pcompile(user_args))
}

var assign_ops = {
  add:  "+",
  sub:  "-",
  mul:  "*",
  div:  "/",
  mod:  "%",
  band: "&",
  bor:  "|",
  bxor: "^",
  lshift: "<<",
  rshift: ">>",
  rrshift: ">>>"
}
;(function(){
  for(var id in assign_ops) {
    var op = assign_ops[id]
    exports[id] = makeOp({
      args: ["array","array","array"],
      body: {args:["a","b","c"],
             body: "a=b"+op+"c"},
      funcName: id
    })
    exports[id+"eq"] = makeOp({
      args: ["array","array"],
      body: {args:["a","b"],
             body:"a"+op+"=b"},
      rvalue: true,
      funcName: id+"eq"
    })
    exports[id+"s"] = makeOp({
      args: ["array", "array", "scalar"],
      body: {args:["a","b","s"],
             body:"a=b"+op+"s"},
      funcName: id+"s"
    })
    exports[id+"seq"] = makeOp({
      args: ["array","scalar"],
      body: {args:["a","s"],
             body:"a"+op+"=s"},
      rvalue: true,
      funcName: id+"seq"
    })
  }
})();

var unary_ops = {
  not: "!",
  bnot: "~",
  neg: "-",
  recip: "1.0/"
}
;(function(){
  for(var id in unary_ops) {
    var op = unary_ops[id]
    exports[id] = makeOp({
      args: ["array", "array"],
      body: {args:["a","b"],
             body:"a="+op+"b"},
      funcName: id
    })
    exports[id+"eq"] = makeOp({
      args: ["array"],
      body: {args:["a"],
             body:"a="+op+"a"},
      rvalue: true,
      count: 2,
      funcName: id+"eq"
    })
  }
})();

var binary_ops = {
  and: "&&",
  or: "||",
  eq: "===",
  neq: "!==",
  lt: "<",
  gt: ">",
  leq: "<=",
  geq: ">="
}
;(function() {
  for(var id in binary_ops) {
    var op = binary_ops[id]
    exports[id] = makeOp({
      args: ["array","array","array"],
      body: {args:["a", "b", "c"],
             body:"a=b"+op+"c"},
      funcName: id
    })
    exports[id+"s"] = makeOp({
      args: ["array","array","scalar"],
      body: {args:["a", "b", "s"],
             body:"a=b"+op+"s"},
      funcName: id+"s"
    })
    exports[id+"eq"] = makeOp({
      args: ["array", "array"],
      body: {args:["a", "b"],
             body:"a=a"+op+"b"},
      rvalue:true,
      count:2,
      funcName: id+"eq"
    })
    exports[id+"seq"] = makeOp({
      args: ["array", "scalar"],
      body: {args:["a","s"],
             body:"a=a"+op+"s"},
      rvalue:true,
      count:2,
      funcName: id+"seq"
    })
  }
})();

var math_unary = [
  "abs",
  "acos",
  "asin",
  "atan",
  "ceil",
  "cos",
  "exp",
  "floor",
  "log",
  "round",
  "sin",
  "sqrt",
  "tan"
]
;(function() {
  for(var i=0; i<math_unary.length; ++i) {
    var f = math_unary[i]
    exports[f] = makeOp({
                    args: ["array", "array"],
                    pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                    body: {args:["a","b"], body:"a=this_f(b)", thisVars:["this_f"]},
                    funcName: f
                  })
    exports[f+"eq"] = makeOp({
                      args: ["array"],
                      pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                      body: {args: ["a"], body:"a=this_f(a)", thisVars:["this_f"]},
                      rvalue: true,
                      count: 2,
                      funcName: f+"eq"
                    })
  }
})();

var math_comm = [
  "max",
  "min",
  "atan2",
  "pow"
]
;(function(){
  for(var i=0; i<math_comm.length; ++i) {
    var f= math_comm[i]
    exports[f] = makeOp({
                  args:["array", "array", "array"],
                  pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                  body: {args:["a","b","c"], body:"a=this_f(b,c)", thisVars:["this_f"]},
                  funcName: f
                })
    exports[f+"s"] = makeOp({
                  args:["array", "array", "scalar"],
                  pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                  body: {args:["a","b","c"], body:"a=this_f(b,c)", thisVars:["this_f"]},
                  funcName: f+"s"
                  })
    exports[f+"eq"] = makeOp({ args:["array", "array"],
                  pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                  body: {args:["a","b"], body:"a=this_f(a,b)", thisVars:["this_f"]},
                  rvalue: true,
                  count: 2,
                  funcName: f+"eq"
                  })
    exports[f+"seq"] = makeOp({ args:["array", "scalar"],
                  pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                  body: {args:["a","b"], body:"a=this_f(a,b)", thisVars:["this_f"]},
                  rvalue:true,
                  count:2,
                  funcName: f+"seq"
                  })
  }
})();

var math_noncomm = [
  "atan2",
  "pow"
]
;(function(){
  for(var i=0; i<math_noncomm.length; ++i) {
    var f= math_noncomm[i]
    exports[f+"op"] = makeOp({
                  args:["array", "array", "array"],
                  pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                  body: {args:["a","b","c"], body:"a=this_f(c,b)", thisVars:["this_f"]},
                  funcName: f+"op"
                })
    exports[f+"ops"] = makeOp({
                  args:["array", "array", "scalar"],
                  pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                  body: {args:["a","b","c"], body:"a=this_f(c,b)", thisVars:["this_f"]},
                  funcName: f+"ops"
                  })
    exports[f+"opeq"] = makeOp({ args:["array", "array"],
                  pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                  body: {args:["a","b"], body:"a=this_f(b,a)", thisVars:["this_f"]},
                  rvalue: true,
                  count: 2,
                  funcName: f+"opeq"
                  })
    exports[f+"opseq"] = makeOp({ args:["array", "scalar"],
                  pre: {args:[], body:"this_f=Math."+f, thisVars:["this_f"]},
                  body: {args:["a","b"], body:"a=this_f(b,a)", thisVars:["this_f"]},
                  rvalue:true,
                  count:2,
                  funcName: f+"opseq"
                  })
  }
})();

exports.any = compile({
  args:["array"],
  pre: EmptyProc,
  body: {args:[{name:"a", lvalue:false, rvalue:true, count:1}], body: "if(a){return true}", localVars: [], thisVars: []},
  post: {args:[], localVars:[], thisVars:[], body:"return false"},
  funcName: "any"
})

exports.all = compile({
  args:["array"],
  pre: EmptyProc,
  body: {args:[{name:"x", lvalue:false, rvalue:true, count:1}], body: "if(!x){return false}", localVars: [], thisVars: []},
  post: {args:[], localVars:[], thisVars:[], body:"return true"},
  funcName: "all"
})

exports.sum = compile({
  args:["array"],
  pre: {args:[], localVars:[], thisVars:["this_s"], body:"this_s=0"},
  body: {args:[{name:"a", lvalue:false, rvalue:true, count:1}], body: "this_s+=a", localVars: [], thisVars: ["this_s"]},
  post: {args:[], localVars:[], thisVars:["this_s"], body:"return this_s"},
  funcName: "sum"
})

exports.prod = compile({
  args:["array"],
  pre: {args:[], localVars:[], thisVars:["this_s"], body:"this_s=1"},
  body: {args:[{name:"a", lvalue:false, rvalue:true, count:1}], body: "this_s*=a", localVars: [], thisVars: ["this_s"]},
  post: {args:[], localVars:[], thisVars:["this_s"], body:"return this_s"},
  funcName: "prod"
})

exports.norm2squared = compile({
  args:["array"],
  pre: {args:[], localVars:[], thisVars:["this_s"], body:"this_s=0"},
  body: {args:[{name:"a", lvalue:false, rvalue:true, count:2}], body: "this_s+=a*a", localVars: [], thisVars: ["this_s"]},
  post: {args:[], localVars:[], thisVars:["this_s"], body:"return this_s"},
  funcName: "norm2squared"
})
  
exports.norm2 = compile({
  args:["array"],
  pre: {args:[], localVars:[], thisVars:["this_s"], body:"this_s=0"},
  body: {args:[{name:"a", lvalue:false, rvalue:true, count:2}], body: "this_s+=a*a", localVars: [], thisVars: ["this_s"]},
  post: {args:[], localVars:[], thisVars:["this_s"], body:"return Math.sqrt(this_s)"},
  funcName: "norm2"
})
  

exports.norminf = compile({
  args:["array"],
  pre: {args:[], localVars:[], thisVars:["this_s"], body:"this_s=0"},
  body: {args:[{name:"a", lvalue:false, rvalue:true, count:4}], body:"if(-a>this_s){this_s=-a}else if(a>this_s){this_s=a}", localVars: [], thisVars: ["this_s"]},
  post: {args:[], localVars:[], thisVars:["this_s"], body:"return this_s"},
  funcName: "norminf"
})

exports.norm1 = compile({
  args:["array"],
  pre: {args:[], localVars:[], thisVars:["this_s"], body:"this_s=0"},
  body: {args:[{name:"a", lvalue:false, rvalue:true, count:3}], body: "this_s+=a<0?-a:a", localVars: [], thisVars: ["this_s"]},
  post: {args:[], localVars:[], thisVars:["this_s"], body:"return this_s"},
  funcName: "norm1"
})

exports.sup = compile({
  args: [ "array" ],
  pre:
   { body: "this_h=-Infinity",
     args: [],
     thisVars: [ "this_h" ],
     localVars: [] },
  body:
   { body: "if(_inline_1_arg0_>this_h)this_h=_inline_1_arg0_",
     args: [{"name":"_inline_1_arg0_","lvalue":false,"rvalue":true,"count":2} ],
     thisVars: [ "this_h" ],
     localVars: [] },
  post:
   { body: "return this_h",
     args: [],
     thisVars: [ "this_h" ],
     localVars: [] }
 })

exports.inf = compile({
  args: [ "array" ],
  pre:
   { body: "this_h=Infinity",
     args: [],
     thisVars: [ "this_h" ],
     localVars: [] },
  body:
   { body: "if(_inline_1_arg0_<this_h)this_h=_inline_1_arg0_",
     args: [{"name":"_inline_1_arg0_","lvalue":false,"rvalue":true,"count":2} ],
     thisVars: [ "this_h" ],
     localVars: [] },
  post:
   { body: "return this_h",
     args: [],
     thisVars: [ "this_h" ],
     localVars: [] }
 })

exports.argmin = compile({
  args:["index","array","shape"],
  pre:{
    body:"{this_v=Infinity;this_i=_inline_0_arg2_.slice(0)}",
    args:[
      {name:"_inline_0_arg0_",lvalue:false,rvalue:false,count:0},
      {name:"_inline_0_arg1_",lvalue:false,rvalue:false,count:0},
      {name:"_inline_0_arg2_",lvalue:false,rvalue:true,count:1}
      ],
    thisVars:["this_i","this_v"],
    localVars:[]},
  body:{
    body:"{if(_inline_1_arg1_<this_v){this_v=_inline_1_arg1_;for(var _inline_1_k=0;_inline_1_k<_inline_1_arg0_.length;++_inline_1_k){this_i[_inline_1_k]=_inline_1_arg0_[_inline_1_k]}}}",
    args:[
      {name:"_inline_1_arg0_",lvalue:false,rvalue:true,count:2},
      {name:"_inline_1_arg1_",lvalue:false,rvalue:true,count:2}],
    thisVars:["this_i","this_v"],
    localVars:["_inline_1_k"]},
  post:{
    body:"{return this_i}",
    args:[],
    thisVars:["this_i"],
    localVars:[]}
})

exports.argmax = compile({
  args:["index","array","shape"],
  pre:{
    body:"{this_v=-Infinity;this_i=_inline_0_arg2_.slice(0)}",
    args:[
      {name:"_inline_0_arg0_",lvalue:false,rvalue:false,count:0},
      {name:"_inline_0_arg1_",lvalue:false,rvalue:false,count:0},
      {name:"_inline_0_arg2_",lvalue:false,rvalue:true,count:1}
      ],
    thisVars:["this_i","this_v"],
    localVars:[]},
  body:{
    body:"{if(_inline_1_arg1_>this_v){this_v=_inline_1_arg1_;for(var _inline_1_k=0;_inline_1_k<_inline_1_arg0_.length;++_inline_1_k){this_i[_inline_1_k]=_inline_1_arg0_[_inline_1_k]}}}",
    args:[
      {name:"_inline_1_arg0_",lvalue:false,rvalue:true,count:2},
      {name:"_inline_1_arg1_",lvalue:false,rvalue:true,count:2}],
    thisVars:["this_i","this_v"],
    localVars:["_inline_1_k"]},
  post:{
    body:"{return this_i}",
    args:[],
    thisVars:["this_i"],
    localVars:[]}
})  

exports.random = makeOp({
  args: ["array"],
  pre: {args:[], body:"this_f=Math.random", thisVars:["this_f"]},
  body: {args: ["a"], body:"a=this_f()", thisVars:["this_f"]},
  funcName: "random"
})

exports.assign = makeOp({
  args:["array", "array"],
  body: {args:["a", "b"], body:"a=b"},
  funcName: "assign" })

exports.assigns = makeOp({
  args:["array", "scalar"],
  body: {args:["a", "b"], body:"a=b"},
  funcName: "assigns" })


exports.equals = compile({
  args:["array", "array"],
  pre: EmptyProc,
  body: {args:[{name:"x", lvalue:false, rvalue:true, count:1},
               {name:"y", lvalue:false, rvalue:true, count:1}], 
        body: "if(x!==y){return false}", 
        localVars: [], 
        thisVars: []},
  post: {args:[], localVars:[], thisVars:[], body:"return true"},
  funcName: "equals"
})



},{"cwise-compiler":5}],18:[function(require,module,exports){
var iota = require("iota-array")
var isBuffer = require("is-buffer")

var hasTypedArrays  = ((typeof Float64Array) !== "undefined")

function compare1st(a, b) {
  return a[0] - b[0]
}

function order() {
  var stride = this.stride
  var terms = new Array(stride.length)
  var i
  for(i=0; i<terms.length; ++i) {
    terms[i] = [Math.abs(stride[i]), i]
  }
  terms.sort(compare1st)
  var result = new Array(terms.length)
  for(i=0; i<result.length; ++i) {
    result[i] = terms[i][1]
  }
  return result
}

function compileConstructor(dtype, dimension) {
  var className = ["View", dimension, "d", dtype].join("")
  if(dimension < 0) {
    className = "View_Nil" + dtype
  }
  var useGetters = (dtype === "generic")

  if(dimension === -1) {
    //Special case for trivial arrays
    var code =
      "function "+className+"(a){this.data=a;};\
var proto="+className+".prototype;\
proto.dtype='"+dtype+"';\
proto.index=function(){return -1};\
proto.size=0;\
proto.dimension=-1;\
proto.shape=proto.stride=proto.order=[];\
proto.lo=proto.hi=proto.transpose=proto.step=\
function(){return new "+className+"(this.data);};\
proto.get=proto.set=function(){};\
proto.pick=function(){return null};\
return function construct_"+className+"(a){return new "+className+"(a);}"
    var procedure = new Function(code)
    return procedure()
  } else if(dimension === 0) {
    //Special case for 0d arrays
    var code =
      "function "+className+"(a,d) {\
this.data = a;\
this.offset = d\
};\
var proto="+className+".prototype;\
proto.dtype='"+dtype+"';\
proto.index=function(){return this.offset};\
proto.dimension=0;\
proto.size=1;\
proto.shape=\
proto.stride=\
proto.order=[];\
proto.lo=\
proto.hi=\
proto.transpose=\
proto.step=function "+className+"_copy() {\
return new "+className+"(this.data,this.offset)\
};\
proto.pick=function "+className+"_pick(){\
return TrivialArray(this.data);\
};\
proto.valueOf=proto.get=function "+className+"_get(){\
return "+(useGetters ? "this.data.get(this.offset)" : "this.data[this.offset]")+
"};\
proto.set=function "+className+"_set(v){\
return "+(useGetters ? "this.data.set(this.offset,v)" : "this.data[this.offset]=v")+"\
};\
return function construct_"+className+"(a,b,c,d){return new "+className+"(a,d)}"
    var procedure = new Function("TrivialArray", code)
    return procedure(CACHED_CONSTRUCTORS[dtype][0])
  }

  var code = ["'use strict'"]

  //Create constructor for view
  var indices = iota(dimension)
  var args = indices.map(function(i) { return "i"+i })
  var index_str = "this.offset+" + indices.map(function(i) {
        return "this.stride[" + i + "]*i" + i
      }).join("+")
  var shapeArg = indices.map(function(i) {
      return "b"+i
    }).join(",")
  var strideArg = indices.map(function(i) {
      return "c"+i
    }).join(",")
  code.push(
    "function "+className+"(a," + shapeArg + "," + strideArg + ",d){this.data=a",
      "this.shape=[" + shapeArg + "]",
      "this.stride=[" + strideArg + "]",
      "this.offset=d|0}",
    "var proto="+className+".prototype",
    "proto.dtype='"+dtype+"'",
    "proto.dimension="+dimension)

  //view.size:
  code.push("Object.defineProperty(proto,'size',{get:function "+className+"_size(){\
return "+indices.map(function(i) { return "this.shape["+i+"]" }).join("*"),
"}})")

  //view.order:
  if(dimension === 1) {
    code.push("proto.order=[0]")
  } else {
    code.push("Object.defineProperty(proto,'order',{get:")
    if(dimension < 4) {
      code.push("function "+className+"_order(){")
      if(dimension === 2) {
        code.push("return (Math.abs(this.stride[0])>Math.abs(this.stride[1]))?[1,0]:[0,1]}})")
      } else if(dimension === 3) {
        code.push(
"var s0=Math.abs(this.stride[0]),s1=Math.abs(this.stride[1]),s2=Math.abs(this.stride[2]);\
if(s0>s1){\
if(s1>s2){\
return [2,1,0];\
}else if(s0>s2){\
return [1,2,0];\
}else{\
return [1,0,2];\
}\
}else if(s0>s2){\
return [2,0,1];\
}else if(s2>s1){\
return [0,1,2];\
}else{\
return [0,2,1];\
}}})")
      }
    } else {
      code.push("ORDER})")
    }
  }

  //view.set(i0, ..., v):
  code.push(
"proto.set=function "+className+"_set("+args.join(",")+",v){")
  if(useGetters) {
    code.push("return this.data.set("+index_str+",v)}")
  } else {
    code.push("return this.data["+index_str+"]=v}")
  }

  //view.get(i0, ...):
  code.push("proto.get=function "+className+"_get("+args.join(",")+"){")
  if(useGetters) {
    code.push("return this.data.get("+index_str+")}")
  } else {
    code.push("return this.data["+index_str+"]}")
  }

  //view.index:
  code.push(
    "proto.index=function "+className+"_index(", args.join(), "){return "+index_str+"}")

  //view.hi():
  code.push("proto.hi=function "+className+"_hi("+args.join(",")+"){return new "+className+"(this.data,"+
    indices.map(function(i) {
      return ["(typeof i",i,"!=='number'||i",i,"<0)?this.shape[", i, "]:i", i,"|0"].join("")
    }).join(",")+","+
    indices.map(function(i) {
      return "this.stride["+i + "]"
    }).join(",")+",this.offset)}")

  //view.lo():
  var a_vars = indices.map(function(i) { return "a"+i+"=this.shape["+i+"]" })
  var c_vars = indices.map(function(i) { return "c"+i+"=this.stride["+i+"]" })
  code.push("proto.lo=function "+className+"_lo("+args.join(",")+"){var b=this.offset,d=0,"+a_vars.join(",")+","+c_vars.join(","))
  for(var i=0; i<dimension; ++i) {
    code.push(
"if(typeof i"+i+"==='number'&&i"+i+">=0){\
d=i"+i+"|0;\
b+=c"+i+"*d;\
a"+i+"-=d}")
  }
  code.push("return new "+className+"(this.data,"+
    indices.map(function(i) {
      return "a"+i
    }).join(",")+","+
    indices.map(function(i) {
      return "c"+i
    }).join(",")+",b)}")

  //view.step():
  code.push("proto.step=function "+className+"_step("+args.join(",")+"){var "+
    indices.map(function(i) {
      return "a"+i+"=this.shape["+i+"]"
    }).join(",")+","+
    indices.map(function(i) {
      return "b"+i+"=this.stride["+i+"]"
    }).join(",")+",c=this.offset,d=0,ceil=Math.ceil")
  for(var i=0; i<dimension; ++i) {
    code.push(
"if(typeof i"+i+"==='number'){\
d=i"+i+"|0;\
if(d<0){\
c+=b"+i+"*(a"+i+"-1);\
a"+i+"=ceil(-a"+i+"/d)\
}else{\
a"+i+"=ceil(a"+i+"/d)\
}\
b"+i+"*=d\
}")
  }
  code.push("return new "+className+"(this.data,"+
    indices.map(function(i) {
      return "a" + i
    }).join(",")+","+
    indices.map(function(i) {
      return "b" + i
    }).join(",")+",c)}")

  //view.transpose():
  var tShape = new Array(dimension)
  var tStride = new Array(dimension)
  for(var i=0; i<dimension; ++i) {
    tShape[i] = "a[i"+i+"]"
    tStride[i] = "b[i"+i+"]"
  }
  code.push("proto.transpose=function "+className+"_transpose("+args+"){"+
    args.map(function(n,idx) { return n + "=(" + n + "===undefined?" + idx + ":" + n + "|0)"}).join(";"),
    "var a=this.shape,b=this.stride;return new "+className+"(this.data,"+tShape.join(",")+","+tStride.join(",")+",this.offset)}")

  //view.pick():
  code.push("proto.pick=function "+className+"_pick("+args+"){var a=[],b=[],c=this.offset")
  for(var i=0; i<dimension; ++i) {
    code.push("if(typeof i"+i+"==='number'&&i"+i+">=0){c=(c+this.stride["+i+"]*i"+i+")|0}else{a.push(this.shape["+i+"]);b.push(this.stride["+i+"])}")
  }
  code.push("var ctor=CTOR_LIST[a.length+1];return ctor(this.data,a,b,c)}")

  //Add return statement
  code.push("return function construct_"+className+"(data,shape,stride,offset){return new "+className+"(data,"+
    indices.map(function(i) {
      return "shape["+i+"]"
    }).join(",")+","+
    indices.map(function(i) {
      return "stride["+i+"]"
    }).join(",")+",offset)}")

  //Compile procedure
  var procedure = new Function("CTOR_LIST", "ORDER", code.join("\n"))
  return procedure(CACHED_CONSTRUCTORS[dtype], order)
}

function arrayDType(data) {
  if(isBuffer(data)) {
    return "buffer"
  }
  if(hasTypedArrays) {
    switch(Object.prototype.toString.call(data)) {
      case "[object Float64Array]":
        return "float64"
      case "[object Float32Array]":
        return "float32"
      case "[object Int8Array]":
        return "int8"
      case "[object Int16Array]":
        return "int16"
      case "[object Int32Array]":
        return "int32"
      case "[object Uint8Array]":
        return "uint8"
      case "[object Uint16Array]":
        return "uint16"
      case "[object Uint32Array]":
        return "uint32"
      case "[object Uint8ClampedArray]":
        return "uint8_clamped"
    }
  }
  if(Array.isArray(data)) {
    return "array"
  }
  return "generic"
}

var CACHED_CONSTRUCTORS = {
  "float32":[],
  "float64":[],
  "int8":[],
  "int16":[],
  "int32":[],
  "uint8":[],
  "uint16":[],
  "uint32":[],
  "array":[],
  "uint8_clamped":[],
  "buffer":[],
  "generic":[]
}

;(function() {
  for(var id in CACHED_CONSTRUCTORS) {
    CACHED_CONSTRUCTORS[id].push(compileConstructor(id, -1))
  }
});

function wrappedNDArrayCtor(data, shape, stride, offset) {
  if(data === undefined) {
    var ctor = CACHED_CONSTRUCTORS.array[0]
    return ctor([])
  } else if(typeof data === "number") {
    data = [data]
  }
  if(shape === undefined) {
    shape = [ data.length ]
  }
  var d = shape.length
  if(stride === undefined) {
    stride = new Array(d)
    for(var i=d-1, sz=1; i>=0; --i) {
      stride[i] = sz
      sz *= shape[i]
    }
  }
  if(offset === undefined) {
    offset = 0
    for(var i=0; i<d; ++i) {
      if(stride[i] < 0) {
        offset -= (shape[i]-1)*stride[i]
      }
    }
  }
  var dtype = arrayDType(data)
  var ctor_list = CACHED_CONSTRUCTORS[dtype]
  while(ctor_list.length <= d+1) {
    ctor_list.push(compileConstructor(dtype, ctor_list.length-1))
  }
  var ctor = ctor_list[d+1]
  return ctor(data, shape, stride, offset)
}

module.exports = wrappedNDArrayCtor

},{"iota-array":11,"is-buffer":12}],19:[function(require,module,exports){
(function (process){
// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

// resolves . and .. elements in a path array with directory names there
// must be no slashes, empty elements, or device names (c:\) in the array
// (so also no leading and trailing slashes - it does not distinguish
// relative and absolute paths)
function normalizeArray(parts, allowAboveRoot) {
  // if the path tries to go above the root, `up` ends up > 0
  var up = 0;
  for (var i = parts.length - 1; i >= 0; i--) {
    var last = parts[i];
    if (last === '.') {
      parts.splice(i, 1);
    } else if (last === '..') {
      parts.splice(i, 1);
      up++;
    } else if (up) {
      parts.splice(i, 1);
      up--;
    }
  }

  // if the path is allowed to go above the root, restore leading ..s
  if (allowAboveRoot) {
    for (; up--; up) {
      parts.unshift('..');
    }
  }

  return parts;
}

// Split a filename into [root, dir, basename, ext], unix version
// 'root' is just a slash, or nothing.
var splitPathRe =
    /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;
var splitPath = function(filename) {
  return splitPathRe.exec(filename).slice(1);
};

// path.resolve([from ...], to)
// posix version
exports.resolve = function() {
  var resolvedPath = '',
      resolvedAbsolute = false;

  for (var i = arguments.length - 1; i >= -1 && !resolvedAbsolute; i--) {
    var path = (i >= 0) ? arguments[i] : process.cwd();

    // Skip empty and invalid entries
    if (typeof path !== 'string') {
      throw new TypeError('Arguments to path.resolve must be strings');
    } else if (!path) {
      continue;
    }

    resolvedPath = path + '/' + resolvedPath;
    resolvedAbsolute = path.charAt(0) === '/';
  }

  // At this point the path should be resolved to a full absolute path, but
  // handle relative paths to be safe (might happen when process.cwd() fails)

  // Normalize the path
  resolvedPath = normalizeArray(filter(resolvedPath.split('/'), function(p) {
    return !!p;
  }), !resolvedAbsolute).join('/');

  return ((resolvedAbsolute ? '/' : '') + resolvedPath) || '.';
};

// path.normalize(path)
// posix version
exports.normalize = function(path) {
  var isAbsolute = exports.isAbsolute(path),
      trailingSlash = substr(path, -1) === '/';

  // Normalize the path
  path = normalizeArray(filter(path.split('/'), function(p) {
    return !!p;
  }), !isAbsolute).join('/');

  if (!path && !isAbsolute) {
    path = '.';
  }
  if (path && trailingSlash) {
    path += '/';
  }

  return (isAbsolute ? '/' : '') + path;
};

// posix version
exports.isAbsolute = function(path) {
  return path.charAt(0) === '/';
};

// posix version
exports.join = function() {
  var paths = Array.prototype.slice.call(arguments, 0);
  return exports.normalize(filter(paths, function(p, index) {
    if (typeof p !== 'string') {
      throw new TypeError('Arguments to path.join must be strings');
    }
    return p;
  }).join('/'));
};


// path.relative(from, to)
// posix version
exports.relative = function(from, to) {
  from = exports.resolve(from).substr(1);
  to = exports.resolve(to).substr(1);

  function trim(arr) {
    var start = 0;
    for (; start < arr.length; start++) {
      if (arr[start] !== '') break;
    }

    var end = arr.length - 1;
    for (; end >= 0; end--) {
      if (arr[end] !== '') break;
    }

    if (start > end) return [];
    return arr.slice(start, end - start + 1);
  }

  var fromParts = trim(from.split('/'));
  var toParts = trim(to.split('/'));

  var length = Math.min(fromParts.length, toParts.length);
  var samePartsLength = length;
  for (var i = 0; i < length; i++) {
    if (fromParts[i] !== toParts[i]) {
      samePartsLength = i;
      break;
    }
  }

  var outputParts = [];
  for (var i = samePartsLength; i < fromParts.length; i++) {
    outputParts.push('..');
  }

  outputParts = outputParts.concat(toParts.slice(samePartsLength));

  return outputParts.join('/');
};

exports.sep = '/';
exports.delimiter = ':';

exports.dirname = function(path) {
  var result = splitPath(path),
      root = result[0],
      dir = result[1];

  if (!root && !dir) {
    // No dirname whatsoever
    return '.';
  }

  if (dir) {
    // It has a dirname, strip trailing slash
    dir = dir.substr(0, dir.length - 1);
  }

  return root + dir;
};


exports.basename = function(path, ext) {
  var f = splitPath(path)[2];
  // TODO: make this comparison case-insensitive on windows?
  if (ext && f.substr(-1 * ext.length) === ext) {
    f = f.substr(0, f.length - ext.length);
  }
  return f;
};


exports.extname = function(path) {
  return splitPath(path)[3];
};

function filter (xs, f) {
    if (xs.filter) return xs.filter(f);
    var res = [];
    for (var i = 0; i < xs.length; i++) {
        if (f(xs[i], i, xs)) res.push(xs[i]);
    }
    return res;
}

// String.prototype.substr - negative index don't work in IE8
var substr = 'ab'.substr(-1) === 'b'
    ? function (str, start, len) { return str.substr(start, len) }
    : function (str, start, len) {
        if (start < 0) start = str.length + start;
        return str.substr(start, len);
    }
;

}).call(this,require('_process'))
},{"_process":20}],20:[function(require,module,exports){
// shim for using process in browser
var process = module.exports = {};

// cached from whatever global is present so that test runners that stub it
// don't break things.  But we need to wrap it in a try catch in case it is
// wrapped in strict mode code which doesn't define any globals.  It's inside a
// function because try/catches deoptimize in certain engines.

var cachedSetTimeout;
var cachedClearTimeout;

function defaultSetTimout() {
    throw new Error('setTimeout has not been defined');
}
function defaultClearTimeout () {
    throw new Error('clearTimeout has not been defined');
}
(function () {
    try {
        if (typeof setTimeout === 'function') {
            cachedSetTimeout = setTimeout;
        } else {
            cachedSetTimeout = defaultSetTimout;
        }
    } catch (e) {
        cachedSetTimeout = defaultSetTimout;
    }
    try {
        if (typeof clearTimeout === 'function') {
            cachedClearTimeout = clearTimeout;
        } else {
            cachedClearTimeout = defaultClearTimeout;
        }
    } catch (e) {
        cachedClearTimeout = defaultClearTimeout;
    }
} ())
function runTimeout(fun) {
    if (cachedSetTimeout === setTimeout) {
        //normal enviroments in sane situations
        return setTimeout(fun, 0);
    }
    // if setTimeout wasn't available but was latter defined
    if ((cachedSetTimeout === defaultSetTimout || !cachedSetTimeout) && setTimeout) {
        cachedSetTimeout = setTimeout;
        return setTimeout(fun, 0);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedSetTimeout(fun, 0);
    } catch(e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't trust the global object when called normally
            return cachedSetTimeout.call(null, fun, 0);
        } catch(e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error
            return cachedSetTimeout.call(this, fun, 0);
        }
    }


}
function runClearTimeout(marker) {
    if (cachedClearTimeout === clearTimeout) {
        //normal enviroments in sane situations
        return clearTimeout(marker);
    }
    // if clearTimeout wasn't available but was latter defined
    if ((cachedClearTimeout === defaultClearTimeout || !cachedClearTimeout) && clearTimeout) {
        cachedClearTimeout = clearTimeout;
        return clearTimeout(marker);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedClearTimeout(marker);
    } catch (e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't  trust the global object when called normally
            return cachedClearTimeout.call(null, marker);
        } catch (e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error.
            // Some versions of I.E. have different rules for clearTimeout vs setTimeout
            return cachedClearTimeout.call(this, marker);
        }
    }



}
var queue = [];
var draining = false;
var currentQueue;
var queueIndex = -1;

function cleanUpNextTick() {
    if (!draining || !currentQueue) {
        return;
    }
    draining = false;
    if (currentQueue.length) {
        queue = currentQueue.concat(queue);
    } else {
        queueIndex = -1;
    }
    if (queue.length) {
        drainQueue();
    }
}

function drainQueue() {
    if (draining) {
        return;
    }
    var timeout = runTimeout(cleanUpNextTick);
    draining = true;

    var len = queue.length;
    while(len) {
        currentQueue = queue;
        queue = [];
        while (++queueIndex < len) {
            if (currentQueue) {
                currentQueue[queueIndex].run();
            }
        }
        queueIndex = -1;
        len = queue.length;
    }
    currentQueue = null;
    draining = false;
    runClearTimeout(timeout);
}

process.nextTick = function (fun) {
    var args = new Array(arguments.length - 1);
    if (arguments.length > 1) {
        for (var i = 1; i < arguments.length; i++) {
            args[i - 1] = arguments[i];
        }
    }
    queue.push(new Item(fun, args));
    if (queue.length === 1 && !draining) {
        runTimeout(drainQueue);
    }
};

// v8 likes predictible objects
function Item(fun, array) {
    this.fun = fun;
    this.array = array;
}
Item.prototype.run = function () {
    this.fun.apply(null, this.array);
};
process.title = 'browser';
process.browser = true;
process.env = {};
process.argv = [];
process.version = ''; // empty string to avoid regexp issues
process.versions = {};

function noop() {}

process.on = noop;
process.addListener = noop;
process.once = noop;
process.off = noop;
process.removeListener = noop;
process.removeAllListeners = noop;
process.emit = noop;

process.binding = function (name) {
    throw new Error('process.binding is not supported');
};

process.cwd = function () { return '/' };
process.chdir = function (dir) {
    throw new Error('process.chdir is not supported');
};
process.umask = function() { return 0; };

},{}],21:[function(require,module,exports){
(function (global,Buffer){
'use strict'

var bits = require('bit-twiddle')
var dup = require('dup')

//Legacy pool support
if(!global.__TYPEDARRAY_POOL) {
  global.__TYPEDARRAY_POOL = {
      UINT8   : dup([32, 0])
    , UINT16  : dup([32, 0])
    , UINT32  : dup([32, 0])
    , INT8    : dup([32, 0])
    , INT16   : dup([32, 0])
    , INT32   : dup([32, 0])
    , FLOAT   : dup([32, 0])
    , DOUBLE  : dup([32, 0])
    , DATA    : dup([32, 0])
    , UINT8C  : dup([32, 0])
    , BUFFER  : dup([32, 0])
  }
}

var hasUint8C = (typeof Uint8ClampedArray) !== 'undefined'
var POOL = global.__TYPEDARRAY_POOL

//Upgrade pool
if(!POOL.UINT8C) {
  POOL.UINT8C = dup([32, 0])
}
if(!POOL.BUFFER) {
  POOL.BUFFER = dup([32, 0])
}

//New technique: Only allocate from ArrayBufferView and Buffer
var DATA    = POOL.DATA
  , BUFFER  = POOL.BUFFER

exports.free = function free(array) {
  if(Buffer.isBuffer(array)) {
    BUFFER[bits.log2(array.length)].push(array)
  } else {
    if(Object.prototype.toString.call(array) !== '[object ArrayBuffer]') {
      array = array.buffer
    }
    if(!array) {
      return
    }
    var n = array.length || array.byteLength
    var log_n = bits.log2(n)|0
    DATA[log_n].push(array)
  }
}

function freeArrayBuffer(buffer) {
  if(!buffer) {
    return
  }
  var n = buffer.length || buffer.byteLength
  var log_n = bits.log2(n)
  DATA[log_n].push(buffer)
}

function freeTypedArray(array) {
  freeArrayBuffer(array.buffer)
}

exports.freeUint8 =
exports.freeUint16 =
exports.freeUint32 =
exports.freeInt8 =
exports.freeInt16 =
exports.freeInt32 =
exports.freeFloat32 = 
exports.freeFloat =
exports.freeFloat64 = 
exports.freeDouble = 
exports.freeUint8Clamped = 
exports.freeDataView = freeTypedArray

exports.freeArrayBuffer = freeArrayBuffer

exports.freeBuffer = function freeBuffer(array) {
  BUFFER[bits.log2(array.length)].push(array)
}

exports.malloc = function malloc(n, dtype) {
  if(dtype === undefined || dtype === 'arraybuffer') {
    return mallocArrayBuffer(n)
  } else {
    switch(dtype) {
      case 'uint8':
        return mallocUint8(n)
      case 'uint16':
        return mallocUint16(n)
      case 'uint32':
        return mallocUint32(n)
      case 'int8':
        return mallocInt8(n)
      case 'int16':
        return mallocInt16(n)
      case 'int32':
        return mallocInt32(n)
      case 'float':
      case 'float32':
        return mallocFloat(n)
      case 'double':
      case 'float64':
        return mallocDouble(n)
      case 'uint8_clamped':
        return mallocUint8Clamped(n)
      case 'buffer':
        return mallocBuffer(n)
      case 'data':
      case 'dataview':
        return mallocDataView(n)

      default:
        return null
    }
  }
  return null
}

function mallocArrayBuffer(n) {
  var n = bits.nextPow2(n)
  var log_n = bits.log2(n)
  var d = DATA[log_n]
  if(d.length > 0) {
    return d.pop()
  }
  return new ArrayBuffer(n)
}
exports.mallocArrayBuffer = mallocArrayBuffer

function mallocUint8(n) {
  return new Uint8Array(mallocArrayBuffer(n), 0, n)
}
exports.mallocUint8 = mallocUint8

function mallocUint16(n) {
  return new Uint16Array(mallocArrayBuffer(2*n), 0, n)
}
exports.mallocUint16 = mallocUint16

function mallocUint32(n) {
  return new Uint32Array(mallocArrayBuffer(4*n), 0, n)
}
exports.mallocUint32 = mallocUint32

function mallocInt8(n) {
  return new Int8Array(mallocArrayBuffer(n), 0, n)
}
exports.mallocInt8 = mallocInt8

function mallocInt16(n) {
  return new Int16Array(mallocArrayBuffer(2*n), 0, n)
}
exports.mallocInt16 = mallocInt16

function mallocInt32(n) {
  return new Int32Array(mallocArrayBuffer(4*n), 0, n)
}
exports.mallocInt32 = mallocInt32

function mallocFloat(n) {
  return new Float32Array(mallocArrayBuffer(4*n), 0, n)
}
exports.mallocFloat32 = exports.mallocFloat = mallocFloat

function mallocDouble(n) {
  return new Float64Array(mallocArrayBuffer(8*n), 0, n)
}
exports.mallocFloat64 = exports.mallocDouble = mallocDouble

function mallocUint8Clamped(n) {
  if(hasUint8C) {
    return new Uint8ClampedArray(mallocArrayBuffer(n), 0, n)
  } else {
    return mallocUint8(n)
  }
}
exports.mallocUint8Clamped = mallocUint8Clamped

function mallocDataView(n) {
  return new DataView(mallocArrayBuffer(n), 0, n)
}
exports.mallocDataView = mallocDataView

function mallocBuffer(n) {
  n = bits.nextPow2(n)
  var log_n = bits.log2(n)
  var cache = BUFFER[log_n]
  if(cache.length > 0) {
    return cache.pop()
  }
  return new Buffer(n)
}
exports.mallocBuffer = mallocBuffer

exports.clearCache = function clearCache() {
  for(var i=0; i<32; ++i) {
    POOL.UINT8[i].length = 0
    POOL.UINT16[i].length = 0
    POOL.UINT32[i].length = 0
    POOL.INT8[i].length = 0
    POOL.INT16[i].length = 0
    POOL.INT32[i].length = 0
    POOL.FLOAT[i].length = 0
    POOL.DOUBLE[i].length = 0
    POOL.UINT8C[i].length = 0
    DATA[i].length = 0
    BUFFER[i].length = 0
  }
}
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {},require("buffer").Buffer)
},{"bit-twiddle":2,"buffer":3,"dup":9}],22:[function(require,module,exports){
"use strict"

function unique_pred(list, compare) {
  var ptr = 1
    , len = list.length
    , a=list[0], b=list[0]
  for(var i=1; i<len; ++i) {
    b = a
    a = list[i]
    if(compare(a, b)) {
      if(i === ptr) {
        ptr++
        continue
      }
      list[ptr++] = a
    }
  }
  list.length = ptr
  return list
}

function unique_eq(list) {
  var ptr = 1
    , len = list.length
    , a=list[0], b = list[0]
  for(var i=1; i<len; ++i, b=a) {
    b = a
    a = list[i]
    if(a !== b) {
      if(i === ptr) {
        ptr++
        continue
      }
      list[ptr++] = a
    }
  }
  list.length = ptr
  return list
}

function unique(list, compare, sorted) {
  if(list.length === 0) {
    return list
  }
  if(compare) {
    if(!sorted) {
      list.sort(compare)
    }
    return unique_pred(list, compare)
  }
  if(!sorted) {
    list.sort()
  }
  return unique_eq(list)
}

module.exports = unique

},{}],23:[function(require,module,exports){
'use strict';

module.exports = {
  printThreshold: 7,
  nFloatingValues: 5
};

},{}],24:[function(require,module,exports){
'use strict';

module.exports = {
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  float32: Float32Array,
  float64: Float64Array,
  array: Array
};

},{}],25:[function(require,module,exports){
'use strict';

module.exports = {
  ValueError: function ValueError () {
    var err = Error.apply(this, arguments);
    err.name = this.constructor.name;
    return err;
  },
  ConfigError: function ConfigError () {
    var err = Error.apply(this, arguments);
    err.name = this.constructor.name;
    return err;
  },
  NotImplementedError: function NotImplementedError () {
    var err = Error.apply(this, arguments);
    err.name = this.constructor.name;
    return err;
  }
};

},{}],26:[function(require,module,exports){
'use strict';

module.exports = function areaSum (h0, w0, H, W, SAT) {
  var x0 = w0 - 1;
  var x1 = w0 + W - 1;
  var y0 = h0 - 1;
  var y1 = h0 + H - 1;
  return (w0 !== 0 && h0 !== 0) ? SAT.selection.get(y0, x0) - SAT.selection.get(y1, x0) - SAT.selection.get(y0, x1) + SAT.selection.get(y1, x1)
    : (w0 === 0 && h0 === 0) ? SAT.selection.get(h0 + H - 1, w0 + W - 1)
      : (w0 === 0) ? -SAT.selection.get(y0, w0 + W - 1) + SAT.selection.get(h0 + H - 1, w0 + W - 1)
        : -SAT.selection.get(y1, x0) + SAT.selection.get(y1, x1);
};

},{}],27:[function(require,module,exports){
'use strict';

var areaSum = require('./area-sum');

module.exports = function areaValue (h0, w0, H, W, SAT) {
  return areaSum(h0, w0, H, W, SAT) / (H * W);
};

},{"./area-sum":26}],28:[function(require,module,exports){
(function (__dirname){
'use strict';
var path = require('path');

var read = require('./read');

var DATA_DIR = path.join(path.resolve(__dirname), '../../data');

function getArray (fileName) {
  return read(path.join(DATA_DIR, fileName));
}

var exports = {};

/**
* @property {NdArray} digit - 28x28 grayscale image with an handwritten digit extracted from MNIST database
*/
Object.defineProperty(exports, 'digit', {
  get: function () {
    return getArray('five.png');
  }
});

/**
* @property {NdArray} five - 28x28 grayscale image with an handwritten digit extracted from MNIST database
*/
Object.defineProperty(exports, 'five', {
  get: function () {
    return getArray('five.png');
  }
});

/**
* @property {NdArray} node - 300x600 COLOR image representing Node.js's logo
*/
Object.defineProperty(exports, 'node', {
  get: function () {
    return getArray('nodejs.png');
  }
});

/**
* @property {NdArray} lena - The standard, yet sometimes controversial Lena test image was scanned from the November 1972 edition of Playboy magazine. From an image processing perspective, this image is useful because it contains smooth, textured, shaded as well as detail areas.
*/
Object.defineProperty(exports, 'lena', {
  get: function () {
    return getArray('lenna.png');
  }
});

/**
* @property {NdArray} lenna - The standard, yet sometimes controversial Lena test image was scanned from the November 1972 edition of Playboy magazine. From an image processing perspective, this image is useful because it contains smooth, textured, shaded as well as detail areas.
*/
Object.defineProperty(exports, 'lenna', {
  get: function () {
    return getArray('lenna.png');
  }
});

/**
* @property {NdArray} moon - This low-contrast image of the surface of the moon is useful for illustrating histogram equalization and contrast stretching.
*/
Object.defineProperty(exports, 'moon', {
  get: function () {
    return getArray('moon.jpg');
  }
});

module.exports = exports;

}).call(this,"/src/images")
},{"./read":32,"path":19}],29:[function(require,module,exports){
'use strict';

var NdArray = require('../ndarray');

module.exports = function flipImage (img) {
  return new NdArray(img.selection.step(null, -1));
};

},{"../ndarray":42}],30:[function(require,module,exports){
'use strict';

/**
 * This callback type is called `imgCallback` and is displayed as a global symbol.
 *
 * @callback imgCallback
 * @param {err} error - if any, null otherwise
 * @param {NdArray} - image represented as a (H, W, [K,]) array, with K the number of color channels. if image is grayscale (or B&W) then image only have two dimensions H and W
 */

module.exports = {
  data: require('./data'),
  read: require('./read'),
  save: require('./save'),
  resize: require('./resize'),
  sat: require('./sat'),
  ssat: require('./ssat'),
  sobel: require('./sobel'),
  scharr: require('./scharr'),
  areaSum: require('./area-sum'),
  areaValue: require('./area-value'),
  rgb2gray: require('./rgb2gray'),
  flip: require('./flip')
};

},{"./area-sum":26,"./area-value":27,"./data":28,"./flip":29,"./read":32,"./resize":33,"./rgb2gray":34,"./sat":35,"./save":36,"./scharr":37,"./sobel":38,"./ssat":39}],31:[function(require,module,exports){
'use strict';


var NdArray = require('../ndarray');

var doCheckIsGrayscale = require('cwise/lib/wrapper')({"args":["array","array","array"],"pre":{"body":"{this_isgray=!0}","args":[],"thisVars":["this_isgray"],"localVars":[]},"body":{"body":"{_inline_82_arg0_===_inline_82_arg1_&&_inline_82_arg1_===_inline_82_arg2_||(this_isgray=!1)}","args":[{"name":"_inline_82_arg0_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_82_arg1_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_82_arg2_","lvalue":false,"rvalue":true,"count":1}],"thisVars":["this_isgray"],"localVars":[]},"post":{"body":"{return this_isgray}","args":[],"thisVars":["this_isgray"],"localVars":[]},"debug":false,"funcName":"doCheckIsGrayscaleCwise","blockSize":64});

module.exports = function isGrayscaleImage (arr) {
  if (arr instanceof NdArray) {
    arr = arr.selection;
  }
  var aShape = arr.shape;
  if (aShape.length === 1) {
    return false;
  }
  if (aShape.length === 2 || aShape.length === 3 && aShape[2] === 1) {
    return true;
  } else if (aShape.length === 3 && (aShape[2] === 3 || aShape[2] === 4)) {
    return doCheckIsGrayscale(arr.pick(null, null, 0), arr.pick(null, null, 1), arr.pick(null, null, 2));
  }
  return false;
};

},{"../ndarray":42,"cwise/lib/wrapper":8}],32:[function(require,module,exports){
'use strict';

/* global HTMLCanvasElement */

var ndarray = require('ndarray');
var NdArray = require('../ndarray');
var errors = require('../errors');
var isGrayscale = require('./is-grayscale');

module.exports = function readImageDom (input) {
  if (input instanceof HTMLCanvasElement) {
    return processCanvas(input);
  } else if (input instanceof HTMLImageElement) {
    return processImg(input);
  } else {
    throw new errors.ValueError('expect input to be either an HTML Canvas or a (loaded) Image');
  }
};

function processCanvas (canvas) {
  var context = canvas.getContext('2d');
  var pixels = context.getImageData(0, 0, canvas.width, canvas.height);

  var shape = [canvas.width, canvas.height, 4];
  var stride = [4, 4 * canvas.width, 1];
  var wxh = ndarray(new Uint8Array(pixels.data), shape, stride, 0);
  var hxw = wxh.transpose(1, 0);

  if (isGrayscale(hxw)) {
    hxw = hxw.pick(null, null, 0);
  }
  return new NdArray(hxw);
}

function processImg (img) {
  var canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  var context = canvas.getContext('2d');
  context.drawImage(img, 0, 0);
  var pixels = context.getImageData(0, 0, img.width, img.height);

  var shape = [img.width, img.height, 4];
  var stride = [4, 4 * img.width, 1];
  var wxh = ndarray(new Uint8Array(pixels.data), shape, stride, 0);
  var hxw = wxh.transpose(1, 0);

  if (isGrayscale(hxw)) {
    hxw = hxw.pick(null, null, 0);
  }
  return new NdArray(hxw);
}

},{"../errors":25,"../ndarray":42,"./is-grayscale":31,"ndarray":18}],33:[function(require,module,exports){
'use strict';

var _ = require('./utils');
var ndarray = require('ndarray');
var NdArray = require('../ndarray');

module.exports = function resizeImageDom (img, height, width) {
  var iShape = img.shape;
  var H = iShape[0];
  var W = iShape[1];
  var K = iShape[2] || 1;
  var originalCanvas = document.createElement('canvas');
  originalCanvas.height = H; originalCanvas.width = W;

  var originalCtx = originalCanvas.getContext('2d');
  var originalImg = originalCtx.createImageData(W, H);
  var err = _.setRawData(img.selection, originalImg.data);
  if (err) { throw err; }

  // compute cropping
  var cfH = H / height;
  var cfW = W / width;
  var cf = Math.min(cfH, cfW);
  var cH = height * cf;
  var cW = width * cf;
  var cdH = (H - cf * height) / 2;
  var cdW = (W - cf * width) / 2;

  originalCtx.putImageData(originalImg, 0, 0);
  originalCtx.drawImage(originalCanvas, cdW, cdH, cW, cH, 0, 0, width, height);

  var resizedImg = originalCtx.getImageData(0, 0, width, height);
  var shape = [width | 0, height | 0, 4];
  var stride = [4, 4 * width | 0, 1];
  var wxh = ndarray(new Uint8Array(resizedImg.data), shape, stride, 0);
  var hxw = wxh.transpose(1, 0);
  if (iShape.length === 2) {
    hxw = hxw.pick(null, null, 0);
  } else if (iShape.length === 3 && K === 1) {
    hxw = hxw.pick(null, null, 0);
  }
  return new NdArray(hxw);
};

},{"../ndarray":42,"./utils":40,"ndarray":18}],34:[function(require,module,exports){
'use strict';


var NdArray = require('../ndarray');
var __ = require('../utils');

// takes ~157ms on a 5000x5000 image
var doRgb2gray = require('cwise/lib/wrapper')({"args":["array","array","array","array"],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_79_arg0_=4899*_inline_79_arg1_+9617*_inline_79_arg2_+1868*_inline_79_arg3_+8192>>14}","args":[{"name":"_inline_79_arg0_","lvalue":true,"rvalue":false,"count":1},{"name":"_inline_79_arg1_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_79_arg2_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_79_arg3_","lvalue":false,"rvalue":true,"count":1}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"rgb2grayCwise","blockSize":64});

/**
 * Compute Grayscale version of an RGB image.
 * @param {NdArray}  img The image in RGB format. In a 3-D array of shape (h, w, 3), or in RGBA format with shape (h, w, 4).
 * @returns {NdArray} The grayscale image, a 3-D array  of shape (h, w, 1).
 */
module.exports = function rgb2gray (img) {
  if (!(img instanceof NdArray)) {
    img = new NdArray(img); // assume it is an ndarray
  }
  var iShape = img.shape;
  var h = iShape[0];
  var w = iShape[1];
  var k = (iShape[2] || 1);
  if (k === 1) {
    return img; // already gray
  }
  var oShape = [h, w];
  var out = new NdArray(new Uint8Array(__.shapeSize(oShape)), oShape);
  var r = img.selection.pick(null, null, 0);
  var g = img.selection.pick(null, null, 1);
  var b = img.selection.pick(null, null, 2);
  doRgb2gray(out.selection, r, g, b);

  return out;
};

},{"../ndarray":42,"../utils":43,"cwise/lib/wrapper":8}],35:[function(require,module,exports){
'use strict';


var NdArray = require('../ndarray');
var rgb2gray = require('./rgb2gray');

var doIntegrate = require('cwise/lib/wrapper')({"args":["array","array","index",{"offset":[-1,-1],"array":0},{"offset":[-1,0],"array":0},{"offset":[0,-1],"array":0}],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_67_arg0_=0!==_inline_67_arg2_[0]&&0!==_inline_67_arg2_[1]?_inline_67_arg1_+_inline_67_arg4_+_inline_67_arg5_-_inline_67_arg3_:0===_inline_67_arg2_[0]&&0===_inline_67_arg2_[1]?_inline_67_arg1_:0===_inline_67_arg2_[0]?_inline_67_arg1_+_inline_67_arg5_:_inline_67_arg1_+_inline_67_arg4_}","args":[{"name":"_inline_67_arg0_","lvalue":true,"rvalue":false,"count":1},{"name":"_inline_67_arg1_","lvalue":false,"rvalue":true,"count":4},{"name":"_inline_67_arg2_","lvalue":false,"rvalue":true,"count":5},{"name":"_inline_67_arg3_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_67_arg4_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_67_arg5_","lvalue":false,"rvalue":true,"count":2}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"doIntegrateBody","blockSize":64});

/**
 * Compute Sum Area Table, also known as the integral of the image
 * @param {NdArray} img
 * @returns {NdArray}
 */
module.exports = function computeSumAreaTable (img) {
  var gray = rgb2gray(img);
  var iShape = gray.shape;
  var iH = iShape[0];
  var iW = iShape[1];
  var out = new NdArray(new Uint32Array(iH * iW), [iH, iW]);

  doIntegrate(out.selection, gray.selection);

  return out;
};

},{"../ndarray":42,"./rgb2gray":34,"cwise/lib/wrapper":8}],36:[function(require,module,exports){
'use strict';

var _ = require('./utils');
var errors = require('../errors');

/**
 * Save image on the given destination
 *
 * @param {NdArray} img
 * @param {HTMLCanvasElement} dest
 */
module.exports = function saveImageDom (img, dest) {
  var iShape = img.shape;
  var iH = iShape[0];
  var iW = iShape[1];
  if (dest instanceof HTMLCanvasElement) {
    var $tmp = document.createElement('canvas');
    $tmp.height = iH; $tmp.width = iW;
    var tmpCtx = $tmp.getContext('2d');
    var originalImg = tmpCtx.createImageData(iW, iH);
    var err = _.setRawData(img.selection, originalImg.data);

    if (err) { throw err; }

    tmpCtx.putImageData(originalImg, 0, 0);
    tmpCtx.drawImage($tmp, iW, iH);
    dest.getContext('2d').drawImage($tmp, 0, 0, iW, iH, 0, 0, dest.width, dest.height);
  } else {
    throw new errors.ValueError('expect input to be either an HTML Canvas or a (loaded) Image');
  }
};

},{"../errors":25,"./utils":40}],37:[function(require,module,exports){
'use strict';


var ops = require('ndarray-ops');
var NdArray = require('../ndarray');
var __ = require('../utils');
var rgb2gray = require('./rgb2gray');

var doScharr = require('cwise/lib/wrapper')({"args":["array","array",{"offset":[-1,-1],"array":1},{"offset":[-1,0],"array":1},{"offset":[-1,1],"array":1},{"offset":[0,-1],"array":1},{"offset":[0,1],"array":1},{"offset":[1,-1],"array":1},{"offset":[1,0],"array":1},{"offset":[1,1],"array":1}],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{var _inline_76_q=3*_inline_76_arg2_+10*_inline_76_arg3_+3*_inline_76_arg4_-3*_inline_76_arg7_-10*_inline_76_arg8_-3*_inline_76_arg9_,_inline_76_s=3*_inline_76_arg2_-3*_inline_76_arg4_+10*_inline_76_arg5_-10*_inline_76_arg6_+3*_inline_76_arg7_-3*_inline_76_arg9_;_inline_76_arg0_=Math.sqrt(_inline_76_s*_inline_76_s+_inline_76_q*_inline_76_q)}","args":[{"name":"_inline_76_arg0_","lvalue":true,"rvalue":false,"count":1},{"name":"_inline_76_arg1_","lvalue":false,"rvalue":false,"count":0},{"name":"_inline_76_arg2_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_76_arg3_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_76_arg4_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_76_arg5_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_76_arg6_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_76_arg7_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_76_arg8_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_76_arg9_","lvalue":false,"rvalue":true,"count":2}],"thisVars":[],"localVars":["_inline_76_q","_inline_76_s"]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"doSobelBody","blockSize":64});

/**
 * Find the edge magnitude using the Scharr transform.
 *
 * @note
 * Take the square root of the sum of the squares of the horizontal and vertical Scharrs to get a magnitude
 * that is somewhat insensitive to direction. The Scharr operator has a better rotation invariance than other
 * edge filters, such as the Sobel operators.
 *
 * @param {NdArray} img
 */
module.exports = function computeScharr (img) {
  var gray = rgb2gray(img);
  var iShape = gray.shape;
  var iH = iShape[0];
  var iW = iShape[1];
  var out = new NdArray(new Float32Array(__.shapeSize(iShape)), iShape);

  doScharr(out.selection, gray.selection);

  // set borders to zero (invalid anyway)
  ops.assigns(out.selection.pick(0, null), 0); // first line
  ops.assigns(out.selection.pick(null, 0), 0); // first col
  ops.assigns(out.selection.pick(iH - 1, null), 0); // last line
  ops.assigns(out.selection.pick(null, iW - 1), 0); // last col

  return out.divide(16 * Math.sqrt(2), false);
};

},{"../ndarray":42,"../utils":43,"./rgb2gray":34,"cwise/lib/wrapper":8,"ndarray-ops":17}],38:[function(require,module,exports){
'use strict';


var ops = require('ndarray-ops');
var NdArray = require('../ndarray');
var __ = require('../utils');
var rgb2gray = require('./rgb2gray');

var doSobel = require('cwise/lib/wrapper')({"args":["array","array",{"offset":[-1,-1],"array":1},{"offset":[-1,0],"array":1},{"offset":[-1,1],"array":1},{"offset":[0,-1],"array":1},{"offset":[0,1],"array":1},{"offset":[1,-1],"array":1},{"offset":[1,0],"array":1},{"offset":[1,1],"array":1}],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{var _inline_73_q=_inline_73_arg2_+2*_inline_73_arg3_+_inline_73_arg4_-_inline_73_arg7_-2*_inline_73_arg8_-_inline_73_arg9_,_inline_73_s=_inline_73_arg2_-_inline_73_arg4_+2*_inline_73_arg5_-2*_inline_73_arg6_+_inline_73_arg7_-_inline_73_arg9_;_inline_73_arg0_=Math.sqrt(_inline_73_s*_inline_73_s+_inline_73_q*_inline_73_q)}","args":[{"name":"_inline_73_arg0_","lvalue":true,"rvalue":false,"count":1},{"name":"_inline_73_arg1_","lvalue":false,"rvalue":false,"count":0},{"name":"_inline_73_arg2_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_73_arg3_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_73_arg4_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_73_arg5_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_73_arg6_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_73_arg7_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_73_arg8_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_73_arg9_","lvalue":false,"rvalue":true,"count":2}],"thisVars":[],"localVars":["_inline_73_q","_inline_73_s"]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"doSobelBody","blockSize":64});

/**
 * Find the edge magnitude using the Sobel transform.
 *
 * @note
 * Take the square root of the sum of the squares of the horizontal and vertical Sobels to get a magnitude that’s somewhat insensitive to direction.
 *
 * The 3x3 convolution kernel used in the horizontal and vertical Sobels is an approximation of the
 * gradient of the image (with some slight blurring since 9 pixels are used to compute the gradient at a given pixel).
 * As an approximation of the gradient, the Sobel operator is not completely rotation-invariant. The Scharr operator should be used for a better rotation invariance.
 * @param {NdArray} img
 */
module.exports = function computeSobel (img) {
  var gray = rgb2gray(img);
  var iShape = gray.shape;
  var iH = iShape[0];
  var iW = iShape[1];

  var out = new NdArray(new Float32Array(__.shapeSize(iShape)), iShape);

  doSobel(out.selection, gray.selection);

  // set borders to zero (invalid anyway)
  ops.assigns(out.selection.pick(0, null), 0); // first line
  ops.assigns(out.selection.pick(null, 0), 0); // first col
  ops.assigns(out.selection.pick(iH - 1, null), 0); // last line
  ops.assigns(out.selection.pick(null, iW - 1), 0); // last col

  return out.divide(4 * Math.sqrt(2), false);
};

},{"../ndarray":42,"../utils":43,"./rgb2gray":34,"cwise/lib/wrapper":8,"ndarray-ops":17}],39:[function(require,module,exports){
'use strict';


var NdArray = require('../ndarray');
var rgb2gray = require('./rgb2gray');

var doIntegrate = require('cwise/lib/wrapper')({"args":["array","array","index",{"offset":[-1,-1],"array":0},{"offset":[-1,0],"array":0},{"offset":[0,-1],"array":0}],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_70_arg0_=0!==_inline_70_arg2_[0]&&0!==_inline_70_arg2_[1]?_inline_70_arg1_*_inline_70_arg1_+_inline_70_arg4_+_inline_70_arg5_-_inline_70_arg3_:0===_inline_70_arg2_[0]&&0===_inline_70_arg2_[1]?_inline_70_arg1_*_inline_70_arg1_:0===_inline_70_arg2_[0]?_inline_70_arg1_*_inline_70_arg1_+_inline_70_arg5_:_inline_70_arg1_*_inline_70_arg1_+_inline_70_arg4_}","args":[{"name":"_inline_70_arg0_","lvalue":true,"rvalue":false,"count":1},{"name":"_inline_70_arg1_","lvalue":false,"rvalue":true,"count":8},{"name":"_inline_70_arg2_","lvalue":false,"rvalue":true,"count":5},{"name":"_inline_70_arg3_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_70_arg4_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_70_arg5_","lvalue":false,"rvalue":true,"count":2}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"doIntegrateBody","blockSize":64});

/**
 * Compute Squared Sum Area Table, also known as the integral of the squared image
 * @param {NdArray} img
 * @returns {NdArray}
 */
module.exports = function computeSquaredSumAreaTable (img) {
  var gray = rgb2gray(img);
  var iShape = gray.shape;
  var iH = iShape[0];
  var iW = iShape[1];

  var out = new NdArray(new Uint32Array(iH * iW), [iH, iW]);

  doIntegrate(out.selection, gray.selection);

  return out;
};

},{"../ndarray":42,"./rgb2gray":34,"cwise/lib/wrapper":8}],40:[function(require,module,exports){
'use strict';

var NdArray = require('../ndarray');

/**
 *
 * @param {NdArray} array
 * @returns {*}
 */
module.exports.getRawData = function getRawData (array) {
  if (array instanceof NdArray) {
    array = array.selection; // faster
  }

  var h;
  var w;
  var ptr = 0;
  var aShape = array.shape;
  var H = aShape[0];
  var W = aShape[1];
  var K = (aShape[2] || 1);
  var data = new Uint8Array(H * W * K);

  if (array.shape.length === 3) {
    if (K === 3) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          data[ptr++] = array.get(h, w, 0);
          data[ptr++] = array.get(h, w, 1);
          data[ptr++] = array.get(h, w, 2);
        }
      }
    } else if (K === 4) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          data[ptr++] = array.get(h, w, 0);
          data[ptr++] = array.get(h, w, 1);
          data[ptr++] = array.get(h, w, 2);
          data[ptr++] = array.get(h, w, 3);
        }
      }
    } else if (K === 1) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          data[ptr++] = array.get(h, w, 0);
        }
      }
    } else {
      return new Error('Incompatible array shape');
    }
  } else if (array.shape.length === 2) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
        data[ptr++] = array.get(h, w);
      }
    }
  } else {
    return new Error('Invalid image');
  }
  return data;
};

module.exports.setRawData = function setRawData (array, data) {
  var h;
  var w;
  var ptr = 0;
  var c;
  var H = array.shape[0];
  var W = array.shape[1];
  var K = array.shape[2] || 1;

  if (array.shape.length === 3) {
    if (K === 3) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          data[ptr++] = array.get(h, w, 0);
          data[ptr++] = array.get(h, w, 1);
          data[ptr++] = array.get(h, w, 2);
          data[ptr++] = 255;
        }
      }
    } else if (K === 4) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          data[ptr++] = array.get(h, w, 0);
          data[ptr++] = array.get(h, w, 1);
          data[ptr++] = array.get(h, w, 2);
          data[ptr++] = array.get(h, w, 3);
        }
      }
    } else if (K === 1) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          c = array.get(h, w, 0);
          data[ptr++] = c;
          data[ptr++] = c;
          data[ptr++] = c;
          data[ptr++] = 255;
        }
      }
    } else {
      return new Error('Incompatible array shape');
    }
  } else if (array.shape.length === 2) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
        c = array.get(h, w);
        data[ptr++] = c;
        data[ptr++] = c;
        data[ptr++] = c;
        data[ptr++] = 255;
      }
    }
  } else {
    return new Error('Invalid image');
  }
};

},{"../ndarray":42}],41:[function(require,module,exports){
'use strict';

var ndarray = require('ndarray');

var ops = require('ndarray-ops');
var ndFFT = require('ndarray-fft');

var CONF = require('./config');
var DTYPES = require('./dtypes');
var NdArray = require('./ndarray');
var _ = require('./utils');
var errors = require('./errors');

function broadcast (shape1, shape2) {
  if (shape1.length === 0 || shape2.length === 0) {
    return;
  }
  var reversed1 = shape1.slice().reverse();
  var reversed2 = shape2.slice().reverse();
  var maxLength = Math.max(shape1.length, shape2.length);
  var outShape = new Array(maxLength);
  for (var i = 0; i < maxLength; i++) {
    if (!reversed1[i] || reversed1[i] === 1) {
      outShape[i] = reversed2[i];
    } else if (!reversed2[i] || reversed2[i] === 1) {
      outShape[i] = reversed1[i];
    } else if (reversed1[i] === reversed2[i]) {
      outShape[i] = reversed1[i];
    } else {
      return;
    }
  }
  return outShape.reverse();
}

/**
* Add arguments, element-wise.
*
* @param {(NdArray|Array|number)} a
* @param {(NdArray|Array|number)} b
* @returns {NdArray}
*/
function add (a, b) {
  return NdArray.new(a).add(b);
}

/**
* Multiply arguments, element-wise.
*
* @param {(Array|NdArray)} a
* @param {(Array|NdArray|number)} b
* @returns {NdArray}
*/
function multiply (a, b) {
  return NdArray.new(a).multiply(b);
}

/**
* Divide `a` by `b`, element-wise.
*
* @param {(Array|NdArray)} a
* @param {(Array|NdArray|number)} b
* @returns {NdArray}
*/
function divide (a, b) {
  return NdArray.new(a).divide(b);
}

/**
* Subtract second argument from the first, element-wise.
*
* @param {(NdArray|Array|number)} a
* @param {(NdArray|Array|number)} b
* @returns {NdArray}
*/
function subtract (a, b) {
  return NdArray.new(a).subtract(b);
}

/**
* Return true if two arrays have the same shape and elements, false otherwise.
* @param {(Array|NdArray)} array1
* @param {(Array|NdArray)} array2
* @returns {boolean}
*/
function equal (array1, array2) {
  return NdArray.new(array1).equal(array2);
}

/**
* Return a copy of the array collapsed into one dimension using row-major order (C-style)

* @param {(Array|NdArray)} array
* @returns {NdArray}
*/
function flatten (array) {
  return NdArray.new(array).flatten();
}

/**
* Gives a new shape to an array without changing its data.
* @param {(Array|NdArray)} array
* @param {Array} shape - The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length
* @returns {NdArray}
*/
function reshape (array, shape) {
  return NdArray.new(array).reshape(shape);
}

/**
* Calculate the exponential of all elements in the input array, element-wise.
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function exp (x) {
  return NdArray.new(x).exp();
}

/**
* Calculate the positive square-root of all elements in the input array, element-wise.
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function sqrt (x) {
  return NdArray.new(x).sqrt();
}

/**
* Raise first array elements to powers from second array, element-wise.
*
* @param {(Array|NdArray|number)} x1
* @param {(Array|NdArray|number)} x2
* @returns {NdArray}
*/
function power (x1, x2) {
  return NdArray.new(x1).pow(x2);
}

/**
* Return the sum of input array elements.
*
* @param {(Array|NdArray|number)} x
* @returns {number}
*/
function sum (x) {
  return NdArray.new(x).sum();
}

/**
* Return the arithmetic mean of input array elements.
*
* @param {(Array|NdArray|number)} x
* @returns {number}
*/
function mean (x) {
  return NdArray.new(x).mean();
}

/**
* Returns the standard deviation, a measure of the spread of a distribution, of the input array elements.
*
* @param {(Array|NdArray|number)} x
* @returns {number}
*/
function std (x) {
  return NdArray.new(x).std();
}

/**
* Return the minimum value of the array
*
* @param {(Array|NdArray|number)} x
* @returns {Number}
*/
function min (x) {
  return NdArray.new(x).min();
}

/**
* Return the maximum value of the array
*
* @param {(Array|NdArray|number)} x
* @returns {Number}
*/
function max (x) {
  return NdArray.new(x).max();
}

/**
* Permute the dimensions of the input array according to the given axes.
*
* @param {(Array|NdArray|number)} x
* @param {(number|...number)} [axes]
* @returns {NdArray}
* @example
*
arr = nj.arange(6).reshape(1,2,3)
// array([[[ 0, 1, 2],
//         [ 3, 4, 5]]])
arr.T
// array([[[ 0],
//         [ 3]],
//        [[ 1],
//         [ 4]],
//        [[ 2],
//         [ 5]]])

arr.transpose(1,0,2)
// array([[[ 0, 1, 2]],
//        [[ 3, 4, 5]]])

*/

function transpose (x, axes) {
  return NdArray.new(x).transpose(axes);
}

/**
* Return the inverse of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function negative (x) {
  return NdArray.new(x).negative();
}

/**
* Return evenly spaced values within a given interval.
*
* @param {int} [start=0] - Start of interval. The interval includes this value.
* @param {int} stop - End of interval. The interval does not include this value.
* @param {int} [step=1] - Spacing between values. The default step size is 1. If step is specified, start must also be given.
* @param {(String|Object)} [dtype=Array] The type of the output array.
*
* @return {NdArray} Array of evenly spaced values.
*/
function arange (start, stop, step, dtype) {
  if (arguments.length === 1) {
    return arange(0, start, 1, undefined);
  } else if (arguments.length === 2 && _.isNumber(stop)) {
    return arange(start, stop, 1, undefined);
  } else if (arguments.length === 2) {
    return arange(0, start, 1, stop);
  } else if (arguments.length === 3 && !_.isNumber(step)) {
    return arange(start, stop, 1, step);
  }
  var result = [];
  var i = 0;
  while (start < stop) {
    result[i++] = start;
    start += step;
  }
  return NdArray.new(result, dtype);
}

/**
* Return a new array of given shape and type, filled with zeros.
*
* @param {(Array|int)} shape - Shape of the new array, e.g., [2, 3] or 2.
* @param {(String|Object)}  [dtype=Array]  The type of the output array.
*
* @return {NdArray} Array of zeros with the given shape and dtype
*/
function zeros (shape, dtype) {
  if (_.isNumber(shape) && shape >= 0) {
    shape = [shape];
  }
  var s = _.shapeSize(shape);
  var T = _.getType(dtype);
  var arr = new NdArray(new T(s), shape);
  if (arr.dtype === 'array') {
    ops.assigns(arr.selection, 0);
  }
  return arr;
}

/**
* Return a new array of given shape and type, filled with ones.
*
* @param {(Array|int)} shape - Shape of the new array, e.g., [2, 3] or 2.
* @param {(String|Object)}  [dtype=Array] - The type of the output array.
*
* @return {NdArray} Array of ones with the given shape and dtype
*/
function ones (shape, dtype) {
  if (_.isNumber(shape) && shape >= 0) {
    shape = [shape];
  }
  var s = _.shapeSize(shape);
  var T = _.getType(dtype);
  var arr = new NdArray(new T(s), shape);
  ops.assigns(arr.selection, 1);
  return arr;
}

/**
* Return a new array of given shape and type, filled with `undefined` values.
*
* @param {(Array|int)} shape - Shape of the new array, e.g., [2, 3] or 2.
* @param {(String|Object)}  [dtype=Array] - The type of the output array.
*
* @return {NdArray} Array of `undefined` values with the given shape and dtype
*/
function empty (shape, dtype) {
  if (_.isNumber(shape) && shape >= 0) {
    shape = [shape];
  }
  var s = _.shapeSize(shape);
  var T = _.getType(dtype);
  return new NdArray(new T(s), shape);
}

/**
* Create an array of the given shape and propagate it with random samples from a uniform distribution over [0, 1].
* @param {number|Array|...number} shape - The dimensions of the returned array, should all be positive integers
* @returns {NdArray}
*/
function random (shape) {
  if (arguments.length === 0) {
    return NdArray.new(Math.random());
  } else if (arguments.length === 1) {
    shape = _.isNumber(shape) ? [shape | 0] : shape;
  } else {
    shape = [].slice.call(arguments);
  }
  var s = _.shapeSize(shape);
  var arr = new NdArray(new Float64Array(s), shape);
  ops.random(arr.selection);
  return arr;
}

/**
* Return the softmax, or normalized exponential, of the input array, element-wise.
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function softmax (x) {
  var e = NdArray.new(x).exp();
  var se = e.sum(); // scalar
  ops.divseq(e.selection, se);
  return e;
}

var doSigmoid = require('cwise/lib/wrapper')({"args":["array","scalar"],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_43_arg0_=_inline_43_arg0_<-30?0:_inline_43_arg0_>30?1:1/(1+Math.exp(-1*_inline_43_arg1_*_inline_43_arg0_))}","args":[{"name":"_inline_43_arg0_","lvalue":true,"rvalue":true,"count":4},{"name":"_inline_43_arg1_","lvalue":false,"rvalue":true,"count":1}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"sigmoidCwise","blockSize":64});

/**
* Return the sigmoid of the input array, element-wise.
* @param {(Array|NdArray|number)} x
* @param {number} [t=1] - stifness parameter
* @returns {NdArray}
*/
function sigmoid (x, t) {
  x = NdArray.new(x).clone();
  t = t || 1;
  doSigmoid(x.selection, t);
  return x;
}

var doClip = require('cwise/lib/wrapper')({"args":["array","scalar","scalar"],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_46_arg0_=Math.min(Math.max(_inline_46_arg1_,_inline_46_arg0_),_inline_46_arg2_)}","args":[{"name":"_inline_46_arg0_","lvalue":true,"rvalue":true,"count":2},{"name":"_inline_46_arg1_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_46_arg2_","lvalue":false,"rvalue":true,"count":1}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"clipCwise","blockSize":64});

/**
* Clip (limit) the values in an array between min and max, element-wise.
*
* @param {(Array|NdArray|number)} x
* @param {number} [min=0]
* @param {number} [max=1]
* @returns {NdArray}
*/
function clip (x, min, max) {
  if (arguments.length === 1) {
    min = 0;
    max = 1;
  } else if (arguments.length === 2) {
    max = 1;
  }
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  doClip(s.selection, min, max);
  return s;
}

var doLeakyRelu = require('cwise/lib/wrapper')({"args":["array","scalar"],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_49_arg0_=Math.max(_inline_49_arg1_*_inline_49_arg0_,_inline_49_arg0_)}","args":[{"name":"_inline_49_arg0_","lvalue":true,"rvalue":true,"count":3},{"name":"_inline_49_arg1_","lvalue":false,"rvalue":true,"count":1}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"leakyReluCwise","blockSize":64});

function leakyRelu (x, alpha) {
  alpha = alpha || 1e-3;
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  doLeakyRelu(s.selection, alpha);
  return s;
}

var doTanh = require('cwise/lib/wrapper')({"args":["array"],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_52_arg0_=(Math.exp(2*_inline_52_arg0_)-1)/(Math.exp(2*_inline_52_arg0_)+1)}","args":[{"name":"_inline_52_arg0_","lvalue":true,"rvalue":true,"count":3}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"tanhCwise","blockSize":64});

/**
* Return hyperbolic tangent of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function tanh (x) {
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  doTanh(s.selection);
  return s;
}

/**
* Return absolute value of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function abs (x) {
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  ops.abseq(s.selection);
  return s;
}

/**
* Return trigonometric cosine of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function cos (x) {
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  ops.coseq(s.selection);
  return s;
}

/**
* Return trigonometric inverse cosine of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function arccos (x) {
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  ops.acoseq(s.selection);
  return s;
}

/**
* Return trigonometric sine of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function sin (x) {
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  ops.sineq(s.selection);
  return s;
}

/**
* Return trigonometric inverse sine of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function arcsin (x) {
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  ops.asineq(s.selection);
  return s;
}

/**
* Return trigonometric tangent of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function tan (x) {
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  ops.taneq(s.selection);
  return s;
}

/**
* Return trigonometric inverse tangent of the input array, element-wise.
*
* @param {(Array|NdArray|number)} x
* @returns {NdArray}
*/
function arctan (x) {
  var s = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  ops.ataneq(s.selection);
  return s;
}

/**
* Dot product of two arrays.
*
* WARNING: supported products are:
*  - matrix dot matrix
*  - vector dot vector
*  - matrix dot vector
*  - vector dot matrix
* @param {(Array|NdArray)} a
* @param {(Array|NdArray)} b
* @returns {NdArray}
*/
function dot (a, b) {
  return NdArray.new(a).dot(b);
}

/**
* Join given arrays along the last axis.
*
* @param {...(Array|NdArray)} arrays
* @returns {NdArray}
*/
function concatenate (arrays) {
  if (arguments.length > 1) {
    arrays = [].slice.call(arguments);
  }
  var i, a;
  for (i = 0; i < arrays.length; i++) {
    a = arrays[i];
    arrays[i] = (a instanceof NdArray) ? a.tolist() : _.isNumber(a) ? [a] : a;
  }
  var m = arrays[0];
  for (i = 1; i < arrays.length; i++) {
    a = arrays[i];
    var mShape = _.getShape(m);
    var aShape = _.getShape(a);
    if (mShape.length !== aShape.length) {
      throw new errors.ValueError('all the input arrays must have same number of dimensions');
    } else if (mShape.length === 1 && aShape.length === 1) {
      m = m.concat(a);
    } else if ((mShape.length === 2 && aShape.length === 2 && mShape[0] === aShape[0]) ||
      (mShape.length === 1 && aShape.length === 2 && mShape[0] === aShape[0]) ||
      (mShape.length === 2 && aShape.length === 1 && mShape[0] === aShape[0])) {
      for (var row = 0; row < mShape[0]; row++) {
        m[row] = m[row].concat(a[row]);
      }
    } else if ((mShape.length === 3 && aShape.length === 3 && mShape[0] === aShape[0] && mShape[1] === aShape[1]) ||
      (mShape.length === 2 && aShape.length === 3 && mShape[0] === aShape[0] && mShape[1] === aShape[1]) ||
      (mShape.length === 3 && aShape.length === 2 && mShape[0] === aShape[0] && mShape[1] === aShape[1])) {
      for (var rowI = 0; rowI < mShape[0]; rowI++) {
        var rowV = new Array(mShape[1]);
        for (var colI = 0; colI < mShape[1]; colI++) {
          rowV[colI] = m[rowI][colI].concat(a[rowI][colI]);
        }
        m[rowI] = rowV;
      }
    } else {
      throw new errors.ValueError('cannot concatenate  "' + mShape + '" with "' + aShape + '"');
    }
  }
  return NdArray.new(m, arrays[0].dtype);
}

/**
* Round an array to the to the nearest integer.
*
* @param {(Array|NdArray)} x
* @returns {NdArray}
*/
function round (x) {
  return NdArray.new(x).round();
}

/**
* Convolve 2 N-dimensionnal arrays
*
* @note: Arrays must have the same dimensions and a must be greater than b.
* @note: The convolution product is only given for points where the signals overlap completely. Values outside the signal boundary have no effect. This behaviour is known as the 'valid' mode.
*
* @param {Array|NdArray} a
* @param {Array|NdArray} b
*/
function convolve (a, b) {
  return NdArray.new(a).convolve(b);
}

/**
* Convolve 2 N-dimensionnal arrays using Fast Fourier Transform (FFT)
*
* @note: Arrays must have the same dimensions and a must be greater than b.
* @note: The convolution product is only given for points where the signals overlap completely. Values outside the signal boundary have no effect. This behaviour is known as the 'valid' mode.
*
* @param {Array|NdArray} a
* @param {Array|NdArray} b
*/
function fftconvolve (a, b) {
  return NdArray.new(a).fftconvolve(b);
}

function fft (x) {
  x = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  var xShape = x.shape;
  var d = xShape.length;
  if (xShape[d - 1] !== 2) {
    throw new errors.ValueError('expect last dimension of the array to have 2 values (for both real and imaginary part)');
  }
  var rPicker = new Array(d);
  var iPicker = new Array(d);
  rPicker[d - 1] = 0;
  iPicker[d - 1] = 1;
  ndFFT(1, x.selection.pick.apply(x.selection, rPicker), x.selection.pick.apply(x.selection, iPicker));
  return x;
}

function ifft (x) {
  x = (x instanceof NdArray) ? x.clone() : NdArray.new(x);
  var xShape = x.shape;
  var d = xShape.length;
  if (xShape[d - 1] !== 2) {
    throw new errors.ValueError('expect last dimension of the array to have 2 values (for both real and imaginary part)');
  }
  var rPicker = new Array(d);
  var iPicker = new Array(d);
  rPicker[d - 1] = 0;
  iPicker[d - 1] = 1;
  ndFFT(-1, x.selection.pick.apply(x.selection, rPicker), x.selection.pick.apply(x.selection, iPicker));
  return x;
}

module.exports = {
  config: CONF,
  dtypes: DTYPES,
  NdArray: NdArray,
  ndarray: ndarray,
  array: NdArray.new,
  arange: arange,
  reshape: reshape,
  zeros: zeros,
  ones: ones,
  empty: empty,
  flatten: flatten,
  random: random,
  softmax: softmax,
  sigmoid: sigmoid,
  leakyRelu: leakyRelu,
  abs: abs,
  arccos: arccos,
  arcsin: arcsin,
  arctan: arctan,
  cos: cos,
  sin: sin,
  tan: tan,
  tanh: tanh,
  clip: clip,
  exp: exp,
  sqrt: sqrt,
  power: power,
  sum: sum,
  mean: mean,
  std: std,
  dot: dot,
  add: add,
  subtract: subtract,
  multiply: multiply,
  divide: divide,
  negative: negative,
  equal: equal,
  max: max,
  min: min,
  concatenate: concatenate,
  transpose: transpose,
  errors: errors,
  broadcast: broadcast,
  round: round,
  convolve: convolve,
  fftconvolve: fftconvolve,
  fft: fft,
  ifft: ifft,
  int8: function (array) { return NdArray.new(array, 'int8'); },
  uint8: function (array) { return NdArray.new(array, 'uint8'); },
  int16: function (array) { return NdArray.new(array, 'int16'); },
  uint16: function (array) { return NdArray.new(array, 'uint16'); },
  int32: function (array) { return NdArray.new(array, 'int32'); },
  uint32: function (array) { return NdArray.new(array, 'uint32'); },
  float32: function (array) { return NdArray.new(array, 'float32'); },
  float64: function (array) { return NdArray.new(array, 'float64'); },
  images: require('./images')
};

},{"./config":23,"./dtypes":24,"./errors":25,"./images":30,"./ndarray":42,"./utils":43,"cwise/lib/wrapper":8,"ndarray":18,"ndarray-fft":13,"ndarray-ops":17}],42:[function(require,module,exports){
'use strict';

var ndarray = require('ndarray');

var ops = require('ndarray-ops');
var gemm = require('ndarray-gemm');
var ndFFT = require('ndarray-fft');
var ndPool = require('typedarray-pool');

var CONF = require('./config');
var errors = require('./errors');
var _ = require('./utils');

/**
* Multidimensional, homogeneous array of fixed-size items
*
* The number of dimensions and items in an array is defined by its shape, which is a tuple of N positive
* integers that specify the sizes of each dimension. The type of items in the array is specified by a separate
* data-type object (dtype), one of which is associated with each NdArray.
* @constructor
*/
var NdArray = function NdArray () {
  if (arguments.length === 1) {
    this.selection = arguments[0];
  } else if (arguments.length === 0) {
    throw new errors.ValueError("Required argument 'data' not found");
  } else {
    this.selection = ndarray.apply(null, arguments);
  }
  /**
  * @property {Number} NdArray#size - Number of elements in the array.
  */
  Object.defineProperty(this, 'size', {
    get: function () {
      return this.selection.size;
    }.bind(this)
  });
  /**
  * The shape of the array
  *
  * @property {Array}
  * @name NdArray#shape
  * @readonly
  */
  Object.defineProperty(this, 'shape', {
    get: function () {
      return this.selection.shape;
    }.bind(this)
  });
  /**
  * Number of array dimensions.
  *
  * @property {Number}
  * @name NdArray#ndim
  * @readonly
  */
  Object.defineProperty(this, 'ndim', {
    get: function () {
      return this.selection.shape.length;
    }.bind(this)
  });
  /**
  * Data-type of the array’s elements.
  *
  * @property {String}
  * @name NdArray#dtype
  * @see {dtypes} for more information
  */
  Object.defineProperty(this, 'dtype', {
    get: function () {
      return this.selection.dtype;
    }.bind(this),
    set: function (dtype) {
      var T = _.getType(dtype);
      if (T !== _.getType(this.dtype)) {
        this.selection = ndarray(new T(this.selection.data), this.selection.shape, this.selection.stride, this.selection.offset);
      }
    }.bind(this)
  });
  /**
  * Permute the dimensions of the array.
  *
  * @property {String}
  * @name NdArray#T
  * @readonly
  */
  Object.defineProperty(this, 'T', {
    get: function () {
      return this.transpose();
    }.bind(this)
  });
};

NdArray.prototype.get = function () {
  var n = arguments.length;
  for (var i = 0; i < n; i++) {
    if (arguments[i] < 0) {
      arguments[i] += this.shape[i];
    }
  }
  return this.selection.get.apply(this.selection, arguments);
};

NdArray.prototype.set = function () {
  return this.selection.set.apply(this.selection, arguments);
};

NdArray.prototype.slice = function () {
  var d = this.ndim;
  var hi = new Array(d);
  var lo = new Array(d);
  var step = new Array(d);
  var tShape = this.shape;

  for (var i = 0; i < d; i++) {
    var arg = arguments[i];
    if (typeof arg === 'undefined') { break; }
    if (arg === null) { continue; }
    if (_.isNumber(arg)) {
      lo[i] = (arg < 0) ? arg + tShape[i] : arg;
      hi[i] = null;
      step[i] = 1;
    } else {
      // assume it is an array
      var start = (arg[0] < 0) ? arg[0] + tShape[i] : arg[0];
      var end = (arg[1] < 0) ? arg[1] + tShape[i] : arg[1];
      lo[i] = end ? start : 0;
      hi[i] = end ? end - start : start;
      step[i] = arg[2] || 1;
    }
  }

  var slo = this.selection.lo.apply(this.selection, lo);
  var shi = slo.hi.apply(slo, hi);
  var sstep = shi.step.apply(shi, step);
  return new NdArray(sstep);
};

/**
* Return a subarray by fixing a particular axis
*
* @param {...(number|null)} axis
* @returns {NdArray}
*
* @example
arr = nj.arange(4*4).reshape(4,4)
// array([[  0,  1,  2,  3],
//        [  4,  5,  6,  7],
//        [  8,  9, 10, 11],
//        [ 12, 13, 14, 15]])

arr.pick(1)
// array([ 4, 5, 6, 7])

arr.pick(null, 1)
// array([  1,  5,  9, 13])
*/
NdArray.prototype.pick = function (axis) {
  return new NdArray(this.selection.pick.apply(this.selection, arguments));
};

/**
* Return a shifted view of the array. Think of it as taking the upper left corner of the image and dragging it inward
*
* @returns {NdArray}
*
* @example
arr = nj.arange(4*4).reshape(4,4)
// array([[  0,  1,  2,  3],
//        [  4,  5,  6,  7],
//        [  8,  9, 10, 11],
//        [ 12, 13, 14, 15]])
arr.lo(1,1)
// array([[  5,  6,  7],
//        [  9, 10, 11],
//        [ 13, 14, 15]])
*/
NdArray.prototype.lo = function () {
  return new NdArray(this.selection.lo.apply(this.selection, arguments));
};

/**
* Return a sliced view of the array.
*
* @returns {NdArray}
*
* @example

arr = nj.arange(4*4).reshape(4,4)
// array([[  0,  1,  2,  3],
//        [  4,  5,  6,  7],
//        [  8,  9, 10, 11],
//        [ 12, 13, 14, 15]])

arr.hi(3,3)
// array([[  0,  1,  2],
//        [  4,  5,  6],
//        [  8,  9, 10]])

arr.lo(1,1).hi(2,2)
// array([[ 5,  6],
//        [ 9, 10]])

*/
NdArray.prototype.hi = function () {
  return new NdArray(this.selection.hi.apply(this.selection, arguments));
};

NdArray.prototype.step = function () {
  return new NdArray(this.selection.step.apply(this.selection, arguments));
};

/**
* Return a copy of the array collapsed into one dimension using row-major order (C-style)
*
* @returns {NdArray}
*/
NdArray.prototype.flatten = function () {
  if (this.ndim === 1) { // already flattened
    return new NdArray(this.selection);
  }
  var T = _.getType(this.dtype);
  var arr = _.flatten(this.tolist(), true);
  if (!(arr instanceof T)) {
    arr = new T(arr);
  }
  return new NdArray(arr, [this.size]);
};

/**
* Gives a new shape to the array without changing its data.
* @param {Array} shape
* @returns {NdArray}
*/
NdArray.prototype.reshape = function (shape) {
  if (arguments.length === 0) {
    throw new errors.ValueError('function takes at least one argument (0 given)');
  }
  if (arguments.length === 1 && _.isNumber(shape)) {
    shape = [shape];
  }
  if (arguments.length > 1) {
    shape = [].slice.call(arguments);
  }
  if (this.size !== _.shapeSize(shape)) {
    throw new errors.ValueError('total size of new array must be unchanged');
  }
  var selfShape = this.selection.shape;
  var selfOffset = this.selection.offset;
  var selfStride = this.selection.stride;
  var selfDim = selfShape.length;
  var d = shape.length;
  var stride;
  var offset;
  var i;
  var sz;
  if (selfDim === d) {
    var sameShapes = true;
    for (i = 0; i < d; ++i) {
      if (selfShape[i] !== shape[i]) {
        sameShapes = false;
        break;
      }
    }
    if (sameShapes) {
      return new NdArray(this.selection.data, selfShape, selfStride, selfOffset);
    }
  } else if (selfDim === 1) {
    // 1d view
    stride = new Array(d);
    for (i = d - 1, sz = 1; i >= 0; --i) {
      stride[i] = sz;
      sz *= shape[i];
    }
    offset = selfOffset;
    for (i = 0; i < d; ++i) {
      if (stride[i] < 0) {
        offset -= (shape[i] - 1) * stride[i];
      }
    }
    return new NdArray(this.selection.data, shape, stride, offset);
  }

  var minDim = Math.min(selfDim, d);
  var areCompatible = true;
  for (i = 0; i < minDim; i++) {
    if (selfShape[i] !== shape[i]) {
      areCompatible = false;
      break;
    }
  }
  if (areCompatible) {
    stride = new Array(d);
    for (i = 0; i < d; i++) {
      stride[i] = selfStride[i] || 1;
    }
    offset = selfOffset;
    return new NdArray(this.selection.data, shape, stride, offset);
  }
  return this.flatten().reshape(shape);
};

/**
* Permute the dimensions of the array.
*
* @param {...number} [axes]
* @returns {NfdArray}
*/
NdArray.prototype.transpose = function (axes) {
  if (arguments.length === 0) {
    var d = this.ndim;
    axes = new Array(d);
    for (var i = 0; i < d; i++) {
      axes[i] = d - i - 1;
    }
  } else if (arguments.length > 1) {
    axes = arguments;
  }
  return new NdArray(this.selection.transpose.apply(this.selection, axes));
};

/**
* Dot product of two arrays.
*
* @param {(Array|NdArray)} x
* @returns {NdArray}
*/
NdArray.prototype.dot = function (x) {
  x = (x instanceof NdArray) ? x : createArray(x, this.dtype);
  var tShape = this.shape;
  var xShape = x.shape;

  if (tShape.length === 2 && xShape.length === 2 && tShape[1] === xShape[0]) { // matrix/matrix
    var T = _.getType(this.dtype);
    var c = new NdArray(new T(tShape[0] * xShape[1]), [tShape[0], xShape[1]]);
    gemm(c.selection, this.selection, x.selection);
    return c;
  } else if (tShape.length === 1 && xShape.length === 2 && tShape[0] === xShape[0]) { // vector/matrix
    return this.reshape([tShape[0], 1]).T.dot(x).reshape(xShape[1]);
  } else if (tShape.length === 2 && xShape.length === 1 && tShape[1] === xShape[0]) { // matrix/vector
    return this.dot(x.reshape([xShape[0], 1])).reshape(tShape[0]);
  } else if (tShape.length === 1 && xShape.length === 1 && tShape[0] === xShape[0]) { // vector/vector
    return this.reshape([tShape[0], 1]).T.dot(x.reshape([xShape[0], 1])).reshape([1]);
  } else {
    throw new errors.ValueError('cannot compute the matrix product of given arrays');
  }
};

/**
* Assign `x` to the array, element-wise.
*
* @param {(NdArray|Array|number)} x
* @param {boolean} [copy=true]
* @returns {NdArray}
*/
NdArray.prototype.assign = function (x, copy) {
  if (arguments.length === 1) {
    copy = true;
  }
  var arr = copy ? this.clone() : this;

  if (_.isNumber(x)) {
    ops.assigns(arr.selection, x);
    return arr;
  }
  x = createArray(x, this.dtype);
  ops.assign(arr.selection, x.selection);
  return arr;
};

/**
* Add `x` to the array, element-wise.
*
* @param {(NdArray|Array|number)} x
* @param {boolean} [copy=true]
* @returns {NdArray}
*/
NdArray.prototype.add = function (x, copy) {
  if (arguments.length === 1) {
    copy = true;
  }
  var arr = copy ? this.clone() : this;

  if (_.isNumber(x)) {
    ops.addseq(arr.selection, x);
    return arr;
  }
  x = createArray(x, this.dtype);
  ops.addeq(arr.selection, x.selection);
  return arr;
};

/**
* Subtract `x` to the array, element-wise.
*
* @param {(NdArray|Array|number)} x
* @param {boolean} [copy=true]
* @returns {NdArray}
*/
NdArray.prototype.subtract = function (x, copy) {
  if (arguments.length === 1) {
    copy = true;
  }
  var arr = copy ? this.clone() : this;

  if (_.isNumber(x)) {
    ops.subseq(arr.selection, x);
    return arr;
  }
  x = createArray(x, this.dtype);
  ops.subeq(arr.selection, x.selection);
  return arr;
};

/**
* Multiply array by `x`, element-wise.
*
* @param {(NdArray|Array|number)} x
* @param {boolean} [copy=true]
* @returns {NdArray}
*/
NdArray.prototype.multiply = function (x, copy) {
  if (arguments.length === 1) {
    copy = true;
  }
  var arr = copy ? this.clone() : this;
  if (_.isNumber(x)) {
    ops.mulseq(arr.selection, x);
    return arr;
  }

  x = createArray(x, this.dtype);
  ops.muleq(arr.selection, x.selection);

  return arr;
};

/**
* Divide array by `x`, element-wise.
*
* @param {(NdArray|Array|number)} x
* @param {boolean} [copy=true]
* @returns {NdArray}
*/
NdArray.prototype.divide = function (x, copy) {
  if (arguments.length === 1) {
    copy = true;
  }
  var arr = copy ? this.clone() : this;
  if (_.isNumber(x)) {
    ops.divseq(arr.selection, x);
    return arr;
  }

  x = createArray(x, this.dtype);
  ops.diveq(arr.selection, x.selection);

  return arr;
};

/**
* Raise array elements to powers from given array, element-wise.
*
* @param {(NdArray|Array|number)} x
* @param {boolean} [copy=true] - set to false to modify the array rather than create a new one
* @returns {NdArray}
*/
NdArray.prototype.pow = function (x, copy) {
  if (arguments.length === 1) { copy = true; }
  var arr = copy ? this.clone() : this;
  if (_.isNumber(x)) {
    ops.powseq(arr.selection, x);
    return arr;
  }

  x = createArray(x, this.dtype);
  ops.poweq(arr.selection, x.selection);
  return arr;
};

/**
* Calculate the exponential of all elements in the array, element-wise.
*
* @param {boolean} [copy=true] - set to false to modify the array rather than create a new one
* @returns {NdArray}
*/
NdArray.prototype.exp = function (copy) {
  if (arguments.length === 0) { copy = true; }
  var arr = copy ? this.clone() : this;
  ops.expeq(arr.selection);
  return arr;
};

/**
* Calculate the positive square-root of all elements in the array, element-wise.
*
* @param {boolean} [copy=true] - set to false to modify the array rather than create a new one
* @returns {NdArray}
*/
NdArray.prototype.sqrt = function (copy) {
  if (arguments.length === 0) { copy = true; }
  var arr = copy ? this.clone() : this;
  ops.sqrteq(arr.selection);
  return arr;
};

/**
* Return the maximum value of the array
*
* @returns {Number}
*/
NdArray.prototype.max = function () {
  if (this.selection.size === 0) {
    return null;
  }
  return ops.sup(this.selection);
};

/**
* Return the minimum value of the array
*
* @returns {Number}
*/
NdArray.prototype.min = function () {
  if (this.selection.size === 0) {
    return null;
  }
  return ops.inf(this.selection);
};

/**
* Sum of array elements.
*
* @returns {number}
*/
NdArray.prototype.sum = function () {
  return ops.sum(this.selection);
};

/**
* Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
*
* @returns {number}
*/
NdArray.prototype.std = function () {
  var squares = this.clone();
  ops.powseq(squares.selection, 2);
  var mean = this.mean();
  var variance = Math.abs(ops.sum(squares.selection) / _.shapeSize(this.shape) - mean * mean);
  return variance > 0 ? Math.sqrt(variance) : 0;
};

/**
* Return the arithmetic mean of array elements.
*
* @returns {number}
*/
NdArray.prototype.mean = function () {
  return ops.sum(this.selection) / _.shapeSize(this.shape);
};

/**
* Converts {NdArray} to a native JavaScript {Array}
*
* @returns {Array}
*/
NdArray.prototype.tolist = function () {
  return unpackArray(this.selection);
};

NdArray.prototype.valueOf = function () {
  return this.tolist();
};
/**
* Stringify the array to make it readable by a human.
*
* @returns {string}
*/
NdArray.prototype.toString = function () {
  var nChars = formatNumber(this.max()).length;

  var reg1 = /\]\,(\s*)\[/g;
  var spacer1 = '],\n$1      [';
  var reg3 = /\]\,(\s+)...\,(\s+)\[/g;
  var spacer3 = '],\n$2       ...\n$2      [';
  var reg2 = /\[\s+\[/g;
  var spacer2 = '[[';

  function formatArray (k, v) {
    if (_.isString(v)) { return v; }
    if (_.isNumber(v)) {
      var s = formatNumber(v);
      return new Array(Math.max(0, nChars - s.length + 2)).join(' ') + s;
    }
    k = k || 0;
    var arr;
    var th = CONF.printThreshold;
    var hth = th / 2 | 0;
    if (v.length > th) {
      arr = [].concat(v.slice(0, hth), [' ...'], v.slice(v.length - hth));
    } else {
      arr = v;
    }
    return new Array(k + 1).join(' ') + '[' + arr.map(function (i, ii) {
      return formatArray(ii === 0 && k === 0 ? 1 : k + 1, i);
    }).join(',') + ']';
  }

  var base = JSON
    .stringify(this.tolist(), formatArray)
    .replace(reg1, spacer1)
    .replace(reg2, spacer2)
    .replace(reg2, spacer2)
    .replace(reg3, spacer3)
    .slice(2, -1);
  switch (this.dtype) {
    case 'array':
      return 'array([' + base + ')';
    default:
      return 'array([' + base + ', dtype=' + this.dtype + ')';
  }
};

/**
* Stringify the array to make it readable in the console, by a human.
*
* @returns {string}
*/
NdArray.prototype.inspect = NdArray.prototype.toString;

/**
* Stringify object to JSON
* @returns {*}
*/
NdArray.prototype.toJSON = function () {
  return JSON.stringify(this.tolist());
};

/**
* Create a full copy of the array
*
* @returns {NdArray}
*/
NdArray.prototype.clone = function () {
  var s = this.selection;
  if (typeof s.data.slice === 'undefined') {
    return new NdArray(ndarray([].slice.apply(s.data), s.shape, s.stride, s.offset)); // for legacy browsers
  }
  return new NdArray(ndarray(s.data.slice(), s.shape, s.stride, s.offset));
};

/**
* Return true if two arrays have the same shape and elements, false otherwise.
* @param {(Array|NdArray)} array
* @returns {boolean}
*/
NdArray.prototype.equal = function (array) {
  array = createArray(array);
  if (this.size !== array.size || this.ndim !== array.ndim) {
    return false;
  }
  var d = this.ndim;
  for (var i = 0; i < d; i++) {
    if (this.shape[i] !== array.shape[i]) {
      return false;
    }
  }

  return ops.all(ops.eqeq(this.selection, array.selection));
};

/**
* Round array to the to the nearest integer.
*
* @param {boolean} [copy=true]
* @returns {NdArray}
*/
NdArray.prototype.round = function (copy) {
  if (arguments.length === 0) {
    copy = true;
  }
  var arr = copy ? this.clone() : this;
  ops.roundeq(arr.selection);
  return arr;
};

/**
* Return the inverse of the array, element-wise.
*
* @returns {NdArray}
*/
NdArray.prototype.negative = function () {
  var c = this.clone();
  ops.neg(c.selection, this.selection);
  return c;
};

NdArray.prototype.iteraxis = function (axis, cb) {
  var shape = this.shape;
  if (axis === -1) {
    axis = shape.length - 1;
  }
  if (axis < 0 || axis > shape.length - 1) {
    throw new errors.ValueError('invalid axis');
  }
  for (var i = 0; i < shape[axis]; i++) {
    var loc = new Array(axis + 1);
    for (var ii = 0; ii < axis + 1; ii++) {
      loc[ii] = (ii === axis) ? i : null;
    }
    var subArr = this.selection.pick.apply(this.selection, loc);
    var xi = createArray(unpackArray(subArr), this.dtype);
    cb(xi, i);
  }
};

var doConjMuleq = require('cwise/lib/wrapper')({"args":["array","array","array","array"],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{var _inline_55_c=_inline_55_arg2_,_inline_55_f=_inline_55_arg3_,_inline_55_i=_inline_55_arg0_,_inline_55_o=_inline_55_arg1_,_inline_55_t=_inline_55_i*(_inline_55_c+_inline_55_f);_inline_55_arg0_=_inline_55_t-_inline_55_f*(_inline_55_i+_inline_55_o),_inline_55_arg1_=_inline_55_t+_inline_55_c*(_inline_55_o-_inline_55_i)}","args":[{"name":"_inline_55_arg0_","lvalue":true,"rvalue":true,"count":2},{"name":"_inline_55_arg1_","lvalue":true,"rvalue":true,"count":2},{"name":"_inline_55_arg2_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_55_arg3_","lvalue":false,"rvalue":true,"count":1}],"thisVars":[],"localVars":["_inline_55_c","_inline_55_f","_inline_55_i","_inline_55_o","_inline_55_t"]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"cwise","blockSize":64});

var doConvolve3x3 = require('cwise/lib/wrapper')({"args":["array","array","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar",{"offset":[-1,-1],"array":1},{"offset":[-1,0],"array":1},{"offset":[-1,1],"array":1},{"offset":[0,-1],"array":1},{"offset":[0,1],"array":1},{"offset":[1,-1],"array":1},{"offset":[1,0],"array":1},{"offset":[1,1],"array":1}],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_58_arg0_=_inline_58_arg11_*_inline_58_arg10_+_inline_58_arg12_*_inline_58_arg9_+_inline_58_arg13_*_inline_58_arg8_+_inline_58_arg14_*_inline_58_arg7_+_inline_58_arg1_*_inline_58_arg6_+_inline_58_arg15_*_inline_58_arg5_+_inline_58_arg16_*_inline_58_arg4_+_inline_58_arg17_*_inline_58_arg3_+_inline_58_arg18_*_inline_58_arg2_}","args":[{"name":"_inline_58_arg0_","lvalue":true,"rvalue":false,"count":1},{"name":"_inline_58_arg1_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg2_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg3_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg4_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg5_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg6_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg7_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg8_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg9_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg10_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg11_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg12_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg13_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg14_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg15_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg16_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg17_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_58_arg18_","lvalue":false,"rvalue":true,"count":1}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"cwise","blockSize":64});

var doConvolve5x5 = require('cwise/lib/wrapper')({"args":["index","array","array","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar","scalar",{"offset":[-2,-2],"array":1},{"offset":[-2,-1],"array":1},{"offset":[-2,0],"array":1},{"offset":[-2,1],"array":1},{"offset":[-2,2],"array":1},{"offset":[-1,-2],"array":1},{"offset":[-1,-1],"array":1},{"offset":[-1,0],"array":1},{"offset":[-1,1],"array":1},{"offset":[-1,2],"array":1},{"offset":[0,-2],"array":1},{"offset":[0,-1],"array":1},{"offset":[0,1],"array":1},{"offset":[0,2],"array":1},{"offset":[1,-2],"array":1},{"offset":[1,-1],"array":1},{"offset":[1,0],"array":1},{"offset":[1,1],"array":1},{"offset":[1,2],"array":1},{"offset":[2,-2],"array":1},{"offset":[2,-1],"array":1},{"offset":[2,0],"array":1},{"offset":[2,1],"array":1},{"offset":[2,2],"array":1}],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{_inline_61_arg1_=_inline_61_arg0_[0]<2||_inline_61_arg0_[1]<2?0:_inline_61_arg28_*_inline_61_arg27_+_inline_61_arg29_*_inline_61_arg26_+_inline_61_arg30_*_inline_61_arg25_+_inline_61_arg31_*_inline_61_arg24_+_inline_61_arg32_*_inline_61_arg23_+_inline_61_arg33_*_inline_61_arg22_+_inline_61_arg34_*_inline_61_arg21_+_inline_61_arg35_*_inline_61_arg20_+_inline_61_arg36_*_inline_61_arg19_+_inline_61_arg37_*_inline_61_arg18_+_inline_61_arg38_*_inline_61_arg17_+_inline_61_arg39_*_inline_61_arg16_+_inline_61_arg2_*_inline_61_arg15_+_inline_61_arg40_*_inline_61_arg14_+_inline_61_arg41_*_inline_61_arg13_+_inline_61_arg42_*_inline_61_arg12_+_inline_61_arg43_*_inline_61_arg11_+_inline_61_arg44_*_inline_61_arg10_+_inline_61_arg45_*_inline_61_arg9_+_inline_61_arg46_*_inline_61_arg8_+_inline_61_arg47_*_inline_61_arg7_+_inline_61_arg48_*_inline_61_arg6_+_inline_61_arg49_*_inline_61_arg5_+_inline_61_arg50_*_inline_61_arg4_+_inline_61_arg51_*_inline_61_arg3_}","args":[{"name":"_inline_61_arg0_","lvalue":false,"rvalue":true,"count":2},{"name":"_inline_61_arg1_","lvalue":true,"rvalue":false,"count":1},{"name":"_inline_61_arg2_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg3_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg4_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg5_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg6_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg7_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg8_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg9_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg10_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg11_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg12_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg13_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg14_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg15_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg16_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg17_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg18_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg19_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg20_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg21_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg22_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg23_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg24_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg25_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg26_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg27_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg28_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg29_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg30_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg31_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg32_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg33_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg34_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg35_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg36_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg37_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg38_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg39_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg40_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg41_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg42_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg43_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg44_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg45_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg46_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg47_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg48_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg49_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg50_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_61_arg51_","lvalue":false,"rvalue":true,"count":1}],"thisVars":[],"localVars":[]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"cwise","blockSize":64});

/**
* Returns the discrete, linear convolution of the array using the given filter.
*
* @note: Arrays must have the same dimensions and `filter` must be smaller than the array.
* @note: The convolution product is only given for points where the signals overlap completely. Values outside the signal boundary have no effect. This behaviour is known as the 'valid' mode.
* @note: Use optimized code for 3x3, 3x3x1, 5x5, 5x5x1 filters, FFT otherwise.
*
* @param {Array|NdArray} filter
*/
NdArray.prototype.convolve = function (filter) {
  filter = NdArray.new(filter);
  var ndim = this.ndim;
  if (ndim !== filter.ndim) {
    throw new errors.ValueError('arrays must have the same dimensions');
  }
  var outShape = new Array(ndim);
  var step = new Array(ndim);
  var ts = this.selection;
  var tShape = this.shape;
  var fs = filter.selection;
  var fShape = filter.shape;

  for (var i = 0; i < ndim; i++) {
    var l = tShape[i] - fShape[i] + 1;
    if (l < 0) {
      throw new errors.ValueError('filter cannot be greater than the array');
    }
    outShape[i] = l;
    step[i] = -1;
  }

  if (ndim === 2 && fShape[0] === 3 && fShape[1] === 3) {
    var out3x3 = new NdArray(new Float32Array(_.shapeSize(tShape)), tShape);
    doConvolve3x3(
      out3x3.selection, // c
      ts, // x
      fs.get(0, 0), // fa
      fs.get(0, 1), // fb
      fs.get(0, 2), // fc
      fs.get(1, 0), // fd
      fs.get(1, 1), // fe
      fs.get(1, 2), // ff
      fs.get(2, 0), // fg
      fs.get(2, 1), // fh
      fs.get(2, 2) // fi
    );
    return out3x3.lo(1, 1).hi(outShape[0], outShape[1]);
  } else if (ndim === 3 && fShape[2] === 1 && tShape[2] === 1 && fShape[0] === 3 && fShape[1] === 3) {
    var out3x3x1 = new NdArray(new Float32Array(_.shapeSize(tShape)), tShape);
    doConvolve3x3(
      out3x3x1.selection.pick(null, null, 0), // c
      ts.pick(null, null, 0), // x
      fs.get(0, 0, 0), // fa
      fs.get(0, 1, 0), // fb
      fs.get(0, 2, 0), // fc
      fs.get(1, 0, 0), // fd
      fs.get(1, 1, 0), // fe
      fs.get(1, 2, 0), // ff
      fs.get(2, 0, 0), // fg
      fs.get(2, 1, 0), // fh
      fs.get(2, 2, 0) // fi
    );
    return out3x3x1.lo(1, 1).hi(outShape[0], outShape[1]);
  } else if (ndim === 2 && fShape[0] === 5 && fShape[1] === 5) {
    var out5x5 = new NdArray(new Float32Array(_.shapeSize(tShape)), tShape);
    doConvolve5x5(
      out5x5.selection, // c
      ts, // x
      fs.get(0, 0), // fa
      fs.get(0, 1), // fb
      fs.get(0, 2), // fc
      fs.get(0, 3), // fd
      fs.get(0, 4), // fe
      fs.get(1, 0), // ff
      fs.get(1, 1), // fg
      fs.get(1, 2), // fh
      fs.get(1, 3), // fi
      fs.get(1, 4), // fj
      fs.get(2, 0), // fk
      fs.get(2, 1), // fl
      fs.get(2, 2), // fm
      fs.get(2, 3), // fn
      fs.get(2, 4), // fo
      fs.get(3, 0), // fp
      fs.get(3, 1), // fq
      fs.get(3, 2), // fr
      fs.get(3, 3), // fs
      fs.get(3, 4), // ft
      fs.get(4, 0), // fu
      fs.get(4, 1), // fv
      fs.get(4, 2), // fw
      fs.get(4, 3), // fx
      fs.get(4, 4) // fy
    );
    return out5x5.lo(2, 2).hi(outShape[0], outShape[1]);
  } else if (ndim === 3 && fShape[2] === 1 && tShape[2] === 1 && fShape[0] === 5 && fShape[1] === 5) {
    var out5x5x1 = new NdArray(new Float32Array(_.shapeSize(tShape)), tShape);
    doConvolve5x5(
      out5x5x1.selection, // c
      ts, // x
      fs.get(0, 0, 0), // fa
      fs.get(0, 1, 0), // fb
      fs.get(0, 2, 0), // fc
      fs.get(0, 3, 0), // fd
      fs.get(0, 4, 0), // fe
      fs.get(1, 0, 0), // ff
      fs.get(1, 1, 0), // fg
      fs.get(1, 2, 0), // fh
      fs.get(1, 3, 0), // fi
      fs.get(1, 4, 0), // fj
      fs.get(2, 0, 0), // fk
      fs.get(2, 1, 0), // fl
      fs.get(2, 2, 0), // fm
      fs.get(2, 3, 0), // fn
      fs.get(2, 4, 0), // fo
      fs.get(3, 0, 0), // fp
      fs.get(3, 1, 0), // fq
      fs.get(3, 2, 0), // fr
      fs.get(3, 3, 0), // fs
      fs.get(3, 4, 0), // ft
      fs.get(4, 0, 0), // fu
      fs.get(4, 1, 0), // fv
      fs.get(4, 2, 0), // fw
      fs.get(4, 3, 0), // fx
      fs.get(4, 4, 0) // fy
    );
    return out5x5x1.lo(2, 2).hi(outShape[0], outShape[1]);
  } else {
    return this.fftconvolve(filter);
  }
};

NdArray.prototype.fftconvolve = function (filter) {
  filter = NdArray.new(filter);

  if (this.ndim !== filter.ndim) {
    throw new errors.ValueError('arrays must have the same dimensions');
  }

  var as = this.selection;
  var bs = filter.selection;
  var d = this.ndim;
  var nsize = 1;
  var nstride = new Array(d);
  var nshape = new Array(d);
  var oshape = new Array(d);
  var i;
  for (i = d - 1; i >= 0; --i) {
    nshape[i] = as.shape[i];
    nstride[i] = nsize;
    nsize *= nshape[i];
    oshape[i] = as.shape[i] - bs.shape[i] + 1;
  }

  var T = _.getType(as.dtype);
  var out = new NdArray(new T(_.shapeSize(oshape)), oshape);
  var outs = out.selection;

  var xT = ndPool.mallocDouble(nsize);
  var x = ndarray(xT, nshape, nstride, 0);
  ops.assigns(x, 0);
  ops.assign(x.hi.apply(x, as.shape), as);

  var yT = ndPool.mallocDouble(nsize);
  var y = ndarray(yT, nshape, nstride, 0);
  ops.assigns(y, 0);

  // FFT x/y
  ndFFT(1, x, y);

  var uT = ndPool.mallocDouble(nsize);
  var u = ndarray(uT, nshape, nstride, 0);
  ops.assigns(u, 0);
  ops.assign(u.hi.apply(u, bs.shape), bs);

  var vT = ndPool.mallocDouble(nsize);
  var v = ndarray(vT, nshape, nstride, 0);
  ops.assigns(v, 0);

  ndFFT(1, u, v);

  doConjMuleq(x, y, u, v);

  ndFFT(-1, x, y);

  var outShape = new Array(d);
  var outOffset = new Array(d);
  var needZeroFill = false;
  for (i = 0; i < d; ++i) {
    if (outs.shape[i] > nshape[i]) {
      needZeroFill = true;
    }
    outOffset[i] = bs.shape[i] - 1;
    outShape[i] = Math.min(outs.shape[i], nshape[i] - outOffset[i]);
  }

  var croppedX;
  if (needZeroFill) {
    ops.assign(outs, 0.0);
  }
  croppedX = x.lo.apply(x, outOffset);
  croppedX = croppedX.hi.apply(croppedX, outShape);
  ops.assign(outs.hi.apply(outs, outShape), croppedX);

  ndPool.freeDouble(xT);
  ndPool.freeDouble(yT);
  ndPool.freeDouble(uT);
  ndPool.freeDouble(vT);
  return out;
};

function createArray (arr, dtype) {
  if (arr instanceof NdArray) { return arr; }
  var T = _.getType(dtype);
  if (_.isNumber(arr)) {
    if (T !== Array) {
      return new NdArray(new T([arr]), [1]);
    } else {
      return new NdArray([arr], [1]);
    }
  }
  var shape = _.getShape(arr);
  if (shape.length > 1) {
    arr = _.flatten(arr, true);
  }
  if (!(arr instanceof T)) {
    arr = new T(arr);
  }
  return new NdArray(arr, shape);
}
NdArray.new = createArray;

module.exports = NdArray;

/*     utils    */
function initNativeArray (shape, i) {
  i = i || 0;
  var c = shape[i] | 0;
  if (c <= 0) { return []; }
  var result = new Array(c);
  var j;
  if (i === shape.length - 1) {
    for (j = 0; j < c; ++j) {
      result[j] = 0;
    }
  } else {
    for (j = 0; j < c; ++j) {
      result[j] = initNativeArray(shape, i + 1);
    }
  }
  return result;
}

var doUnpack = require('cwise/lib/wrapper')({"args":["array","scalar","index"],"pre":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"body":{"body":"{var _inline_64_a,_inline_64_e=_inline_64_arg1_;for(_inline_64_a=0;_inline_64_a<_inline_64_arg2_.length-1;++_inline_64_a)_inline_64_e=_inline_64_e[_inline_64_arg2_[_inline_64_a]];_inline_64_e[_inline_64_arg2_[_inline_64_arg2_.length-1]]=_inline_64_arg0_}","args":[{"name":"_inline_64_arg0_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_64_arg1_","lvalue":false,"rvalue":true,"count":1},{"name":"_inline_64_arg2_","lvalue":false,"rvalue":true,"count":4}],"thisVars":[],"localVars":["_inline_64_a","_inline_64_e"]},"post":{"body":"{}","args":[],"thisVars":[],"localVars":[]},"debug":false,"funcName":"unpackCwise","blockSize":64});

function unpackArray (arr) {
  var result = initNativeArray(arr.shape, 0);
  doUnpack(arr, result);
  return result;
}

function formatNumber (v) {
  return String(Number((v || 0).toFixed(CONF.nFloatingValues)));
}

},{"./config":23,"./errors":25,"./utils":43,"cwise/lib/wrapper":8,"ndarray":18,"ndarray-fft":13,"ndarray-gemm":15,"ndarray-ops":17,"typedarray-pool":21}],43:[function(require,module,exports){
'use strict';
var DTYPES = require('./dtypes');

function isNumber (value) {
  return typeof value === 'number';
}
function isString (value) {
  return typeof value === 'string';
}
function isFunction (value) {
  return typeof value === 'function';
}

function baseFlatten (array, isDeep, result) {
  result = result || [];
  var index = -1;
  var length = array.length;

  while (++index < length) {
    var value = array[index];
    if (isNumber(value)) {
      result[result.length] = value;
    } else if (isDeep) {
      // Recursively flatten arrays (susceptible to call stack limits).
      baseFlatten(value, isDeep, result);
    } else {
      result.push(value);
    }
  }

  return result;
}

function shapeSize (shape) {
  var s = 1;
  for (var i = 0; i < shape.length; i++) {
    s *= shape[i];
  }
  return s;
}

function getType (dtype) {
  return isFunction(dtype) ? dtype : (DTYPES[dtype] || Array);
}

function _dim (x) {
  var ret = [];
  while (typeof x === 'object') { ret.push(x.length); x = x[0]; }
  return ret;
}

function getShape (array) {
  var y, z;
  if (typeof array === 'object') {
    y = array[0];
    if (typeof y === 'object') {
      z = y[0];
      if (typeof z === 'object') {
        return _dim(array);
      }
      return [array.length, y.length];
    }
    return [array.length];
  }
  return [];
}

// function haveSameShape (shape1, shape2) {
//   if (shapeSize(shape1) !== shapeSize(shape2) || shape1.length !== shape2.length) {
//     return false;
//   }
//   var d = shape1.length;
//   for (var i = 0; i < d; i++) {
//     if (shape1[i] !== shape2[i]) {
//       return false;
//     }
//   }
//   return true;
// }

module.exports = {
  isNumber: isNumber,
  isString: isString,
  isFunction: isFunction,
  flatten: baseFlatten,
  shapeSize: shapeSize,
  getType: getType,
  getShape: getShape
};

},{"./dtypes":24}]},{},[41])(41)
});