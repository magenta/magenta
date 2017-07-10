// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
/**
 * Author: David Ha <hadavid@google.com>
 *
 * @fileoverview Sketch-RNN Javascript Inference Library
 */

/**
 * Location of JSON models used for sketch-rnn-js
 */
var SketchRNNConfig = {
  BaseURL: "https://storage.googleapis.com/quickdraw-models/sketchRNN/models/"
};

/**
 * Tool to load simplify lines of a sketch using RDP Algorithm
 */
var DataTool = {};

(function (global) {
  "use strict";

  function simplify_line(V, tolerance) {
    // from https://gist.github.com/adammiller/826148
    // V ... [[x1,y1],[x2,y2],...] polyline
    // tol  ... approximation tolerance
    // ==============================================
    // Copyright 2002, softSurfer (www.softsurfer.com)
    // This code may be freely used and modified for any purpose
    // providing that this copyright notice is included with it.
    // SoftSurfer makes no warranty for this code, and cannot be held
    // liable for any real or imagined damage resulting from its use.
    // Users of this code must verify correctness for their application.
    // http://softsurfer.com/Archive/algorithm_0205/algorithm_0205.htm

    var tol=2.0;
    if (typeof(tolerance) === "number") {
      tol = tolerance;
    }

    var sum = function(u,v) {return [u[0]+v[0], u[1]+v[1]];}
    var diff = function(u,v) {return [u[0]-v[0], u[1]-v[1]];}
    var prod = function(u,v) {return [u[0]*v[0], u[1]*v[1]];}
    var dot = function(u,v) {return u[0]*v[0] + u[1]*v[1];}
    var norm2 = function(v) {return v[0]*v[0] + v[1]*v[1];}
    var norm = function(v) {return Math.sqrt(norm2(v));}
    var d2 = function(u,v) {return norm2(diff(u,v));}
    var d = function(u,v) {return norm(diff(u,v));}

    var simplifyDP = function( tol, v, j, k, mk ) {
      //  This is the Douglas-Peucker recursive simplification routine
      //  It just marks vertices that are part of the simplified polyline
      //  for approximating the polyline subchain v[j] to v[k].
      //  mk[] ... array of markers matching vertex array v[]
      if (k <= j+1) { // there is nothing to simplify
        return;
      }
      // check for adequate approximation by segment S from v[j] to v[k]
      var maxi = j;          // index of vertex farthest from S
      var maxd2 = 0;         // distance squared of farthest vertex
      var tol2 = tol * tol;  // tolerance squared
      var S = [v[j], v[k]];  // segment from v[j] to v[k]
      var u = diff(S[1], S[0]);   // segment direction vector
      var cu = norm2(u,u);     // segment length squared
      // test each vertex v[i] for max distance from S
      // compute using the Feb 2001 Algorithm's dist_Point_to_Segment()
      // Note: this works in any dimension (2D, 3D, ...)
      var  w;           // vector
      var Pb;                // point, base of perpendicular from v[i] to S
      var b, cw, dv2;        // dv2 = distance v[i] to S squared
      for (var i=j+1; i<k; i++) {
        // compute distance squared
        w = diff(v[i], S[0]);
        cw = dot(w,u);
        if ( cw <= 0 ) {
          dv2 = d2(v[i], S[0]);
        } else if ( cu <= cw ) {
          dv2 = d2(v[i], S[1]);
        } else {
          b = cw / cu;
          Pb = [S[0][0]+b*u[0], S[0][1]+b*u[1]];
          dv2 = d2(v[i], Pb);
        }
        // test with current max distance squared
        if (dv2 <= maxd2) {
          continue;
        }
        // v[i] is a new max vertex
        maxi = i;
        maxd2 = dv2;
      }
      if (maxd2 > tol2) {      // error is worse than the tolerance
        // split the polyline at the farthest vertex from S
        mk[maxi] = 1;      // mark v[maxi] for the simplified polyline
        // recursively simplify the two subpolylines at v[maxi]
        simplifyDP( tol, v, j, maxi, mk );  // polyline v[j] to v[maxi]
        simplifyDP( tol, v, maxi, k, mk );  // polyline v[maxi] to v[k]
      }
      // else the approximation is OK, so ignore intermediate vertices
      return;
    }

    var n = V.length;
    var sV = [];
    var i, k, m, pv;               // misc counters
    var tol2 = tol * tol;          // tolerance squared
    var vt = [];                       // vertex buffer, points
    var mk = [];                       // marker buffer, ints

    // STAGE 1.  Vertex Reduction within tolerance of prior vertex cluster
    vt[0] = V[0];              // start at the beginning
    for (i=k=1, pv=0; i<n; i++) {
      if (d2(V[i], V[pv]) < tol2) {
        continue;
      }
      vt[k++] = V[i];
      pv = i;
    }
    if (pv < n-1) {
      vt[k++] = V[n-1];      // finish at the end
    }

    // STAGE 2.  Douglas-Peucker polyline simplification
    mk[0] = mk[k-1] = 1;       // mark the first and last vertices
    simplifyDP( tol, vt, 0, k-1, mk );

    // copy marked vertices to the output simplified polyline
    for (i=m=0; i<k; i++) {
      if (mk[i]) {
        sV[m++] = vt[i];
      }
    }
    return sV;
  }

  /**
   * Clean wrapper method to use RDP function.
   */
  function simplify_lines(lines) {
    var result = [];
    var tolerance = 2.0;
    for (var i=0;i<lines.length;i++) {
      result.push(simplify_line(lines[i], tolerance));
    }
    return result;
  };

  /**
   * convert from polylines to stroke-5 format that sketch-rnn uses
   */
  function lines_to_strokes(raw_data) {
    var x, y;
    var px=0, py=0;
    var dx, dy;
    var pon, poff;
    var stroke = [];
    var i, j;
    var len;
    var p;
    for (i=0;i<raw_data.length;i++) {
      len = raw_data[i].length;
      if (len > 1) {
        for (j=0;j<len;j++) {
          p = raw_data[i][j];
          x = p[0];
          y = p[1];
          if (j === len-1) {
            poff = 1;
            pon = 0;
          } else {
            poff = 0;
            pon = 1;
          }
          dx = x - px;
          dy = y - py;
          px = x;
          py = y;
          stroke.push([dx, dy, pon, poff, 0]);
        }
      }
    }
    stroke.push([0, 0, 0, 0, 1]);
    return stroke.slice(1);
  };

  /**
   * convert from stroke-5 format to polylines
   */
  function line_to_stroke(line, last_point) {
    var pon, poff;
    var stroke = [];
    var len;
    var p;
    var dx, dy;
    var x, y;
    var px, py;
    var j;
    px = last_point[0];
    py = last_point[1];
    len = line.length;
    if (len > 1) {
      for (j=0;j<len;j++) {
        p = line[j];
        //x = p.x;
        //y = p.y;
        x = p[0];
        y = p[1];
        if (j === len-1) {
          poff = 1;
          pon = 0;
        } else {
          poff = 0;
          pon = 1;
        }
        dx = x - px;
        dy = y - py;
        px = x;
        py = y;
        stroke.push([dx, dy, pon, poff, 0]);
      }
    }

    return stroke;

  };

  global.lines_to_strokes = lines_to_strokes;
  global.simplify_line = simplify_line;
  global.simplify_lines = simplify_lines;
  global.line_to_stroke = line_to_stroke;

})(DataTool);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    // usable in browser
  } else {
    module.exports = lib; // in nodejs
  }
})(DataTool);


/**
 * Simple tool to load JSON model files dynamically
 */
var ModelImporter = {};

(function(global) {
  "use strict";

  /**
   * load a given JSON model file dynamically
   */
  function loadJSON(filename, callback) {
    var xobj = new XMLHttpRequest();
        xobj.overrideMimeType("application/json");
        xobj.open('GET', filename, true);
        // Replace 'my_data' with the path to your file
        xobj.onreadystatechange = function () {
          if (xobj.readyState == 4 && xobj.status == "200") {
            // Required use of an anonymous callback
            // as .open() will NOT return a value but simply returns undefined in asynchronous mode
            callback(xobj.responseText);
          }
    };
    xobj.send(null);
  }

  // settings
  var init_model_data;
  var model_data_archive = [];
  var model_url = SketchRNNConfig.BaseURL;

  /**
   * assume given a parsed model model_raw_data, put it inside the model archives.
   * must do this at the beginning.
   */
  var set_init_model = function(model_raw_data) {
    init_model_data = JSON.parse(model_raw_data);
    model_data_archive[init_model_data[0].name+"_"+init_model_data[0].mode] = model_raw_data;
  };

  /**
   * Return's the current model selected in the database of possible models
   */
  var get_model_data = function() {
    return init_model_data;
  };

  /**
   * Have capability to override where to download the models from (if not on googlecloud)
   */
  var set_model_url = function(url) {
    model_url = url;
  };

  /**
   * Change the model to another class (ie, from ant to frog).
   */
  var change_model = function(model, class_name, class_type, call_back, do_not_cache) {
    if (model && typeof(class_name) === "string" && typeof(class_type) === "string") {
      var model_name = class_name + "." + class_type;
      console.log("attempting to load "+model_name);
      if (model_data_archive[model_name]) {
        console.log("changing with cached "+model_name);
        var new_model = new SketchRNN(JSON.parse(model_data_archive[model_name]));
        if (call_back) {
          call_back(new_model);
        }
        return;
      } else { // not cached
        var cache_model_mode = true;
        if (typeof do_not_cache === "undefined") {
          cache_model_mode = true;
        } else {
          if (do_not_cache) {
            cache_model_mode = false;
          } else {
            cache_model_mode = true;
          }
        }
        console.log("loading "+model_name+" dynamically");
        loadJSON(model_url+model_name+'.json', function(response) {
          console.log("callback from json load");
          // Parse JSON string into object
          var result = JSON.parse(response);
          if (cache_model_mode) {
            console.log("caching the model.");
            model_data_archive[model_name] = response; // cache it
          } else {
            console.log("not caching the model.");
          }
          var new_model = new SketchRNN(result);
          // model.load_model(model_data_archive[model_name]);
          if (call_back) {
            call_back(new_model);
          }
          return;
         });
      }
    }
  };

  global.get_model_data = get_model_data;
  global.set_init_model = set_init_model;
  global.set_model_url = set_model_url;
  global.change_model = change_model;

})(ModelImporter);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(ModelImporter);

/**
 * Internal LSTM class used by sketch-rnn
 * @class
 * @constructor
 */
function LSTMCell(num_units, input_size, Wxh, Whh, bias) {
  this.num_units = num_units;
  this.input_size = input_size;
  this.Wxh = Wxh;
  this.Whh = Whh;
  this.bias = bias;
  this.forget_bias = 1.0;
  this.Wfull=nj.concatenate([Wxh.T, Whh.T]).T;
}
LSTMCell.prototype.zero_state = function() {
  return [nj.zeros(this.num_units), nj.zeros(this.num_units)];
};
LSTMCell.prototype.forward = function(x, h, c) {
  var concat = nj.concatenate([x, h]);
  var hidden = nj.add(nj.dot(concat, this.Wfull), this.bias);
  var num_units = this.num_units;
  var forget_bias = this.forget_bias;

  var i=nj.sigmoid(hidden.slice([0*num_units, 1*num_units]));
  var g=nj.tanh(hidden.slice([1*num_units, 2*num_units]));
  var f=nj.sigmoid(nj.add(hidden.slice([2*num_units, 3*num_units]), forget_bias));
  var o=nj.sigmoid(hidden.slice([3*num_units, 4*num_units]));

  var new_c = nj.add(nj.multiply(c, f), nj.multiply(g, i));
  var new_h = nj.multiply(nj.tanh(new_c), o);

  return [new_h, new_c];
};
LSTMCell.prototype.encode = function(sequence) {
  var x;
  var state = this.zero_state();
  var h = state[0];
  var c = state[1];
  var N = sequence.length;
  for (var i=0;i<N;i++) {
    x = nj.array(sequence[i]);
    state = this.forward(x, h, c);
    h = state[0];
    c = state[1];
  }
  return h;
};

/**
 * The sketch-rnn model. Please see README.md for documentation.
 * @class
 */
function SketchRNN(model_raw_data) {
  "use strict";

  // settings
  var info;
  var dimensions;
  var num_blobs;
  var weights;
  var max_weight;
  var N_mixture;

  var max_seq_len;

  var pixel_factor;
  var scale_factor;

  // model variables:
  var enc_fw_lstm_W_xh, enc_fw_lstm_W_hh, enc_fw_lstm_bias, enc_bw_lstm_W_xh,enc_bw_lstm_W_hh,enc_bw_lstm_bias,enc_mu_w,enc_mu_b,enc_sigma_w,enc_sigma_b,enc_w,enc_b,dec_output_w,dec_output_b,dec_lstm_W_xh,dec_lstm_W_hh,dec_lstm_bias;
  var dec_num_units, dec_input_size, enc_num_units, enc_input_size, z_size;
  var enc_fw_lstm, enc_bw_lstm, dec_lstm;

  /**
   * deals with decompressing b64 models to float arrays.
   */
  function string_to_uint8array(b64encoded) {
    var u8 = new Uint8Array(atob(b64encoded).split("").map(function(c) {
      return c.charCodeAt(0); }));
    return u8;
  }
  function uintarray_to_string(u8) {
    var s = "";
    for (var i = 0, len = u8.length; i < len; i++) {
      s += String.fromCharCode(u8[i]);
    }
    var b64encoded = btoa(s);
    return b64encoded;
  };
  function string_to_array(s) {
    var u = string_to_uint8array(s);
    var result = new Int16Array(u.buffer);
    return result;
  };
  function array_to_string(a) {
    var u = new Uint8Array(a.buffer);
    var result = uintarray_to_string(u);
    return result;
  };

  // Random numbers util (from https://github.com/karpathy/recurrentjs)
  var return_v = false;
  var v_val = 0.0;
  function gaussRandom() {
    if(return_v) {
      return_v = false;
      return v_val;
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  }
  function randf(a, b) { return Math.random()*(b-a)+a; };
  function randi(a, b) { return Math.floor(Math.random()*(b-a)+a); };
  function randn(mu, std){ return mu+gaussRandom()*std; };
  // from http://www.math.grin.edu/~mooret/courses/math336/bivariate-normal.html
  function birandn(mu1, mu2, std1, std2, rho) {
    var z1 = randn(0, 1);
    var z2 = randn(0, 1);
    var x = Math.sqrt(1-rho*rho)*std1*z1 + rho*std1*z2 + mu1;
    var y = std2*z2 + mu2;
    return [x, y];
  };

  /**
   * sets the internal scale factor of this model (usually ~ 100)
   */
  function set_scale_factor(scale) {
    scale_factor = scale;
    this.scale_factor = scale_factor;
  };

  /**
   * sets the internal EXTRA factor of this model (pixel to model space)
   */
  function set_pixel_factor(scale) {
    // for best effect, set to 1.0 for d3 or paper.js, 2.0 for p5.js
    pixel_factor = scale;
    scale_factor = info.scale_factor / pixel_factor;
    this.scale_factor = scale_factor;
  };

  /**
   * loads a JSON-parsed model in its specified format.
   */
  function load_model(model_raw_data) {
    "use strict";
    var i, j;

    info = model_raw_data[0];
    dimensions = model_raw_data[1];
    num_blobs = dimensions.length;
    var weightsIn = model_raw_data[2];
    weights = Array(weightsIn.length)
    max_weight = 10.0;
    N_mixture=20;

    max_seq_len = info.max_seq_len;

    pixel_factor = 2.0; // maybe should be 1.0 for non-retina screens.
    scale_factor = info.scale_factor / pixel_factor;  // divide by 2 for macbooks?

    for (i=0;i<num_blobs;i++) {
      weights[i] = nj.array(new Float32Array(string_to_array(weightsIn[i])), 'float32');
      weights[i] = weights[i].divide(32767);
      weights[i] = weights[i].multiply(max_weight);
      if(dimensions[i].length == 2) {
        var d = dimensions[i];
        var d1 = d[0], d2 = d[1];
        weights[i] = weights[i].reshape(d1, d2);
      }
    }

    if(info.mode === 2 || info.mode === "gen") { // 0 or 1 - vae, 2 - gen
      dec_output_w = weights[0];
      dec_output_b = weights[1];
      dec_lstm_W_xh = weights[2];
      dec_lstm_W_hh = weights[3];
      dec_lstm_bias = weights[4];
    } else {
      enc_fw_lstm_W_xh = weights[0];
      enc_fw_lstm_W_hh = weights[1];
      enc_fw_lstm_bias = weights[2];
      enc_bw_lstm_W_xh = weights[3];
      enc_bw_lstm_W_hh = weights[4];
      enc_bw_lstm_bias = weights[5];
      enc_mu_w = weights[6];
      enc_mu_b = weights[7];
      enc_sigma_w = weights[8];
      enc_sigma_b = weights[9];
      enc_w = weights[10];
      enc_b = weights[11];
      dec_output_w = weights[12];
      dec_output_b = weights[13];
      dec_lstm_W_xh = weights[14];
      dec_lstm_W_hh = weights[15];
      dec_lstm_bias = weights[16];
      enc_num_units = enc_fw_lstm_W_hh.shape[0];
      enc_input_size = enc_fw_lstm_W_xh.shape[0];
      z_size = enc_w.shape[0];

      enc_fw_lstm = new LSTMCell(enc_num_units, enc_input_size, enc_fw_lstm_W_xh, enc_fw_lstm_W_hh, enc_fw_lstm_bias);
      enc_bw_lstm = new LSTMCell(enc_num_units, enc_input_size, enc_bw_lstm_W_xh, enc_bw_lstm_W_hh, enc_bw_lstm_bias);
    }

    dec_num_units = dec_lstm_W_hh.shape[0];
    dec_input_size = dec_lstm_W_xh.shape[0];

    dec_lstm = new LSTMCell(dec_num_units, dec_input_size, dec_lstm_W_xh, dec_lstm_W_hh, dec_lstm_bias);
    console.log("loading model...");
    console.log("class="+info.name);
    console.log("version="+info.version);
    console.log("model type="+info.mode);
    console.log("train size="+info.name);
    console.log("scale factor="+Math.round(1000*info.scale_factor)/1000);
    console.log("reconst loss="+Math.round(1000*info.r_loss)/1000);
    console.log("kl loss="+Math.round(1000*info.kl_loss)/1000);
    console.log("max seq len="+info.max_seq_len);
    console.log("training sample size="+info.training_size);
  };

  /**
   * returns a vector of IID zero mean unit variance gaussian values.
   */
  function random_latent_vector() {
    var z = new Array(z_size);
    for (var i=0;i<z_size;i++) {
      z[i] = gaussRandom();
    }
    return z;
  };

  /**
   * element-wise hyperbolic tangent.
   */
  function tanh(z) {
    var y = new Array(z.length);
    for (var i=0;i<z.length;i++) {
      y[i] = Math.tanh(z[i]);
    }
    return y;
  };

  /**
   * returns a vector of IID zero mean unit variance gaussian values.
   */
  function random_normal_vector() {
    // the same as random_latent_vector. in the future, random_latent_vector may be non-gaussian.
    var z = new Array(z_size);
    for (var i=0;i<z_size;i++) {
      z[i] = gaussRandom();
    }
    return z;
  };

  /**
   * encodes a drawing into a latent vector
   */
  function encode(sequence, temperature) {
    // returns latent vector given vector image. optional temperature parameter to have stochastic embeddings.

    var temp=1.0;
    if (typeof(temperature) === "number") {
      temp = temperature;
    };

    var forward_sequence = [];
    var reverse_sequence = [];
    var i;
    var s;
    var N = sequence.length;
    for (i=0;i<N;i++) {
      s = [sequence[i][0]/scale_factor, sequence[i][1]/scale_factor, sequence[i][2], sequence[i][3], sequence[i][4]];
      forward_sequence.push(s);
    }
    for (i=N-1;i>=0;i--) {
      s = [forward_sequence[i][0], forward_sequence[i][1], forward_sequence[i][2], forward_sequence[i][3], forward_sequence[i][4]];
      reverse_sequence.push(s);
    }
    var output_fw = enc_fw_lstm.encode(forward_sequence);
    var output_bw = enc_bw_lstm.encode(reverse_sequence);
    var output = nj.concatenate([output_fw, output_bw]);
    var mu = nj.add(nj.dot(output, enc_mu_w),enc_mu_b);
    // optimization:
    if (temp > 0) {
      var presig = nj.add(nj.dot(output, enc_sigma_w),enc_sigma_b);
      var sigma = nj.sqrt(nj.exp(presig));
      var eps = nj.multiply(nj.array(random_normal_vector(), 'float32'), temp);
      var z = nj.add(mu, nj.multiply(eps, sigma));
    } else {
      var z = mu;
    }
    return z.tolist();
  };

  /**
   * don't encode to latent vector, but only up to mu and sigma (for speed-hungry applications)
   */
  function encode_to_mu_sigma(sequence) {
    var forward_sequence = [];
    var reverse_sequence = [];
    var i;
    var s;
    var N = sequence.length;
    for (i=0;i<N;i++) {
      s = [sequence[i][0]/scale_factor, sequence[i][1]/scale_factor, sequence[i][2], sequence[i][3], sequence[i][4]];
      forward_sequence.push(s);
    }
    for (i=N-1;i>=0;i--) {
      s = [forward_sequence[i][0], forward_sequence[i][1], forward_sequence[i][2], forward_sequence[i][3], forward_sequence[i][4]];
      reverse_sequence.push(s);
    }
    var output_fw = enc_fw_lstm.encode(forward_sequence);
    var output_bw = enc_bw_lstm.encode(reverse_sequence);
    var output = nj.concatenate([output_fw, output_bw]);
    var mu = nj.add(nj.dot(output, enc_mu_w),enc_mu_b);

    var presig = nj.add(nj.dot(output, enc_sigma_w),enc_sigma_b);
    var sigma = nj.sqrt(nj.exp(presig));

    return [mu, sigma];
  };

  /**
   * given a fixed mu/sigma, sample a v=latent vector.
   */
  function encode_from_mu_sigma(mu, sigma, temperature) {
    // returns random z given mu, sigma, temperature (2nd part of encode function)
    // sometimes we want to create a large list of random z's given the same encoded image
    // this is an optimization.
    var temp=1.0;
    if (typeof(temperature) === "number") {
      temp = temperature;
    };
    var eps = nj.multiply(nj.array(random_normal_vector(), 'float32'), temp);
    var z = nj.add(mu, nj.multiply(eps, sigma));
    return z.tolist();
  };

  /**
   * given a latent vector, get the initial states of the LSTM (h and c)
   */
  function get_init_state_from_latent_vector(y) {

    if (y.constructor != Array) {
      console.log("error, the argument passed into decode must be a Javascript normal Array type.");
      y = Array.prototype.slice.call(y);
    };

    var z = nj.array(y);

    /**
     * for backward compatibility.
     */
    if (this.info.version < 6) {
      z = nj.tanh(z); // for older versions, clip the latent variable.
    }

    var init_state = nj.tanh(nj.add(nj.dot(z, enc_w), enc_b));

    var c = init_state.slice([0, dec_num_units]).clone();
    var h = init_state.slice([dec_num_units, 2*dec_num_units]).clone();

    return [h, c];
  };

  /**
   * given a latent vector, sample a vector image using full vae-model,
   */
  function decode(y, temperature, softmax_temperature) {
    if (y.constructor != Array) {
      console.log("error, the argument passed into decode must be a Javascript normal Array type.");
      y = Array.prototype.slice.call(y);
    };

    var temp=0.01;
    if (typeof(temperature) === "number") {
      temp = temperature;
    }
    var softmax_temp= 0.5+temp*0.5;
    if (typeof(softmax_temperature) === "number") {
      softmax_temp = softmax_temperature;
    }

    var z = nj.array(y);

    var init_state = nj.tanh(nj.add(nj.dot(z, enc_w), enc_b));

    var c = init_state.slice([0, dec_num_units]).clone();
    var h = init_state.slice([dec_num_units, 2*dec_num_units]).clone();

    var rnn_state;
    var dx, dy, pen_down, pen_up, pen_end;
    var pdf;
    var x = nj.array([0, 0, 0, 0, 0]);
    var result = [];
    var lstm_input;

    for(var i=0;i<max_seq_len;i++) {
      lstm_input = nj.concatenate([x, z]);
      rnn_state = dec_lstm.forward(lstm_input, h, c);
      pdf = get_pdf(rnn_state);
      [dx, dy, pen_down, pen_up, pen_end] = sample(pdf, temp, softmax_temp);
      result.push([dx, dy, pen_down, pen_up, pen_end]);
      if (pen_end === 1) {
        return result;
      }
      x = nj.array([dx/scale_factor, dy/scale_factor, pen_down, pen_up, pen_end]);
      h = rnn_state[0];
      c = rnn_state[1];
    }
    result.push([0, 0, 0, 0, 1]);
    return result;

  };

  /**
   * given a latent vector, sample vector image using decoder-only ("gen") models.
   */
  function generate(temperature, softmax_temperature) {
    // returns a random, complete.

    var temp=0.65;
    if (typeof(temperature) === "number") {
      temp = temperature;
    }
    var softmax_temp = 0.5+temp*0.5;
    if (typeof(softmax_temperature) === "number") {
      softmax_temp = softmax_temperature;
    }

    var init_state = zero_state();
    var h = init_state[0];
    var c = init_state[1];

    var rnn_state;
    var dx, dy, pen_down, pen_up, pen_end;
    var pdf;
    var x = nj.array([0, 0, 0, 0, 0]);
    var result = [];
    var lstm_input;

    for(var i=0;i<max_seq_len;i++) {
      lstm_input = x;
      rnn_state = dec_lstm.forward(lstm_input, h, c);
      pdf = get_pdf(rnn_state);
      [dx, dy, pen_down, pen_up, pen_end] = sample(pdf, temp, softmax_temp);
      result.push([dx, dy, pen_down, pen_up, pen_end]);
      if (pen_end === 1) {
        return result;
      }
      x = nj.array([dx/scale_factor, dy/scale_factor, pen_down, pen_up, pen_end]);
      h = rnn_state[0];
      c = rnn_state[1];
    }
    result.push([0, 0, 0, 0, 1]);
    return result;

  };

  /**
   * make a new copy of a drawing
   */
  function copy_drawing(sequence, maximum_length) {
    // a copy of the drawing with an optional maximum_length param.
    var result = [];
    var sx = 0, sy = 0;
    var max_len = sequence.length;
    if (typeof maximum_length === "number") {
      max_len = maximum_length;
    }
    var N = Math.min(sequence.length, max_len);
    for (var i=0;i<N;i++) {
      result.push([sequence[i][0], sequence[i][1], sequence[i][2], sequence[i][3], sequence[i][4]]);
    }
    var seq_len = result.length;
    if (sequence[seq_len-1][4] != 1) {
      // if we cut off the drawing, just make the last point end-of-line, and add end-of-drawing token.
      sequence[seq_len-1][2] = 0;
      sequence[seq_len-1][3] = 1;
      sequence[seq_len-1][4] = 0;
      result.push([0, 0, 0, 0, 1]);
    }

    return result;
  };

  /**
   * scale the drawing to be inside a fixed box.
   */
  function scale_drawing(sequence, scale) {
    // returns vector image that is scaled so that the maximum size is the same as the scale parameter.
    // scale parameter is in pixels.
    var result = [];
    var sx = 0, sy = 0;
    var max_x = -10000000, min_x = 10000000;
    var max_y = -10000000, min_y = 10000000;
    for (var i=0;i<sequence.length;i++) {
      if (sequence[i][4] === 1) {
        continue;
      }
      sx += sequence[i][0];
      sy += sequence[i][1];
      max_x = Math.max(max_x, sx);
      min_x = Math.min(min_x, sx);
      max_y = Math.max(max_y, sy);
      min_y = Math.min(min_y, sy);
    }

    var max_size = Math.max(max_x - min_x, max_y - min_y);
    var factor = scale / max_size;

    for (var i=0;i<sequence.length;i++) {
      result.push([sequence[i][0]*factor, sequence[i][1]*factor, sequence[i][2], sequence[i][3], sequence[i][4]]);
    }

    return result;
  };

  /**
   * scale the drawing by a fixed factor
   */
  function scale_drawing_by_factor(sequence, factor) {
    // returns vector image that is scaled by multiplictive factor.
    // scale parameter is in pixels.
    var result = [];

    for (var i=0;i<sequence.length;i++) {
      result.push([sequence[i][0]*factor, sequence[i][1]*factor, sequence[i][2], sequence[i][3], sequence[i][4]]);
    }

    return result;
  };

  /**
   * center the drawing so that the starting point will be the center of the image.
   */
  function center_drawing(sequence) {
    // returns vector image that is centered at the starting point.
    var result = [];
    var sx = 0, sy = 0;
    var max_x = -10000000, min_x = 10000000;
    var max_y = -10000000, min_y = 10000000;
    for (var i=0;i<sequence.length;i++) {
      if (sequence[i][4] === 1) {
        result.push([0, 0, 0, 0, 1]);
        continue;
      }
      sx += sequence[i][0];
      sy += sequence[i][1];
      max_x = Math.max(max_x, sx);
      min_x = Math.min(min_x, sx);
      max_y = Math.max(max_y, sy);
      min_y = Math.min(min_y, sy);
      result.push([sequence[i][0], sequence[i][1], sequence[i][2], sequence[i][3], sequence[i][4]]);
    }

    var cx = -(max_x + min_x) / 2.0;
    var cy = -(max_y + min_y) / 2.0;

    result.unshift([cx, cy, 1, 0, 0]);
    return result;
  };

  /**
   * return an empty state of sketch-rnn
   */
  function zero_state() {
    return dec_lstm.zero_state();
  };

  /**
   * returns a copy of the rnn state (for multiple predictions given the same starting point)
   */
  function copy_state(state) {
    var h = state[0].clone();
    var c = state[1].clone();
    return [h, c];
  };

  /**
   * Initially the input is zero for "gen" type models
   */
  function zero_input() {
    return [0, 0, 0, 0, 0];
  };

  /**
   * update the rnn with input x, state s, and optional latent vector y.
   */
  function update(x, s, y) {
    // y is an optional vector parameter, used for conditional mode only.

    var x_ = nj.array([x[0]/scale_factor, x[1]/scale_factor, x[2], x[3], x[4]]);

    var lstm_input, rnn_state;

    if (y) {
      var z = nj.array(y);
      lstm_input = nj.concatenate([x_, z]);
    } else {
      lstm_input = x_;
    }

    rnn_state = dec_lstm.forward(lstm_input, s[0], s[1]);

    return rnn_state;
  };

  /**
   * Gets the parameters of the mixture density distribution for the next point
   */
  function get_pdf(s) {
    var h = s[0];
    var NOUT = N_mixture;
    var z=nj.add(nj.dot(h, dec_output_w), dec_output_b);
    var z_pen_logits = z.slice([0, 3]);
    var z_pi = z.slice([3+NOUT*0, 3+NOUT*1]);
    var z_mu1 = z.slice([3+NOUT*1, 3+NOUT*2]);
    var z_mu2 = z.slice([3+NOUT*2, 3+NOUT*3]);
    var z_sigma1 = nj.exp(z.slice([3+NOUT*3, 3+NOUT*4]));
    var z_sigma2 = nj.exp(z.slice([3+NOUT*4, 3+NOUT*5]));
    var z_corr = nj.tanh(z.slice([3+NOUT*5, 3+NOUT*6]));
    z_pen_logits = nj.subtract(z_pen_logits, z_pen_logits.max());
    var z_pen = nj.softmax(z_pen_logits);
    z_pi = nj.subtract(z_pi, z_pi.max());
    z_pi = nj.softmax(z_pi);

    return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen];
  };

  /**
   * sample from a categorial distribution
   */
  function sample_softmax(z_sample) {
    var x = randf(0, 1);
    var N = z_sample.shape[0];
    var accumulate = 0;
    var i;
    for (i=0;i<N;i++) {
      accumulate += z_sample.get(i);
      if (accumulate >= x) {
        return i;
      }
    }
    console.log('error sampling pi index');
    return -1;
  };

  /**
   * sample from a binary distribution
   */
  function sample_binary(z_eos) {
    // eos = 1 if random.random() < o_eos[0][0] else 0
    var eos = 0;
    if (randf(0, 1) < z_eos.get(0)) {
      eos = 1;
    }
    return eos;
  }

  /**
   * adjust the temperature of a categorial distribution
   */
  function adjust_temp(z_old, temp) {
    var z = nj.array(z_old);
    var i;
    var x;
    //console.log("before="+z_old.get(0));
    for (i=z.shape[0]-1;i>=0;i--) {
      x = z.get(i);
      x = Math.log(x) / temp;
      z.set(i, x);
    }
    x = z.max();
    z = nj.subtract(z, x);
    z = nj.exp(z);
    x = z.sum();
    z = nj.divide(z, x);
    //console.log("after="+z.get(0));
    return z;
  };

  /**
   * samples the next point of the sketch given pdf parameters and optional temperature params
   */
  function sample(z, temperature, softmax_temperature) {
    // z is [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen]
    // returns [x, y, eos]
    // optional softmax_temperature
    var temp=0.65;
    if (typeof(temperature) === "number") {
      temp = temperature;
    }
    var softmax_temp = 0.5+temp*0.5;
    if (typeof(softmax_temperature) === "number") {
      softmax_temp = softmax_temperature;
    }
    var z_0 = adjust_temp(z[0], softmax_temp);
    var z_6 = adjust_temp(z[6], softmax_temp);
    var idx = sample_softmax(z_0);
    var mu1 = z[1].get(idx);
    var mu2 = z[2].get(idx);
    var sigma1 = z[3].get(idx)*Math.sqrt(temp);
    var sigma2 = z[4].get(idx)*Math.sqrt(temp);
    var corr = z[5].get(idx);
    var pen_idx = sample_softmax(z_6);
    var penstate = [0, 0, 0];
    penstate[pen_idx] = 1;
    var delta = birandn(mu1, mu2, sigma1, sigma2, corr);
    return [delta[0]*scale_factor, delta[1]*scale_factor, penstate[0], penstate[1], penstate[2]];
  }

  load_model(model_raw_data);

  function get_info() {
    return this.info;
  };

  this.zero_state = zero_state;
  this.zero_input = zero_input;
  this.copy_state = copy_state;
  this.update = update;
  this.get_pdf = get_pdf;
  this.randf = randf;
  this.randi = randi;
  this.randn = randn;
  this.birandn = birandn;
  this.tanh = tanh;
  this.sample = sample;
  this.scale_factor = scale_factor;
  this.info = info;
  this.max_seq_len = info.max_seq_len;
  this.name = info.name;
  this.mode = info.mode;
  this.set_scale_factor = set_scale_factor;
  this.set_pixel_factor = set_pixel_factor;
  this.encode = encode;
  this.decode = decode;
  this.generate = generate;
  this.random_latent_vector = random_latent_vector;
  this.scale_drawing = scale_drawing;
  this.scale_drawing_by_factor = scale_drawing_by_factor;
  this.center_drawing = center_drawing;
  this.z_size = z_size;
  this.get_info = get_info;
  this.get_init_state_from_latent_vector = get_init_state_from_latent_vector;
  this.encode_to_mu_sigma = encode_to_mu_sigma;
  this.encode_from_mu_sigma = encode_from_mu_sigma;
  this.copy_drawing = copy_drawing;

}
