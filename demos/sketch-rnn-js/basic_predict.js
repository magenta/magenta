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
 * @fileoverview Basic p5.js sketch to show how to use sketch-rnn
 * to complete an unfinished sketch.
 */

var strokes=[[-4,0,1,0,0],[-15,9,1,0,0],[-10,17,1,0,0],[-1,28,1,0,0],[14,13,1,0,0],[12,4,1,0,0],[22,1,1,0,0],[14,-11,1,0,0],[5,-12,1,0,0],[2,-19,1,0,0],[-12,-23,1,0,0],[-13,-7,1,0,0],[-14,-1,0,1,0]];

// sketch_rnn model
var rnn_model;
var rnn_model_data;
var temperature = 0.25;
var min_sequence_length = 5;

var model_pdf; // store all the parameters of a mixture-density distribution
var model_state;
var model_prev_pen;
var model_x, model_y;

// variables for the sketch input interface.
var start_x, start_y;
var end_x, end_y;

// UI
var screen_width, screen_height;
var line_width = 1.0;
var line_color, predict_line_color;

// dom
var model_sel;

var draw_example = function(example, start_x, start_y, line_color) {
  var i;
  var x=start_x, y=start_y;
  var dx, dy;
  var pen_down, pen_up, pen_end;
  var prev_pen = [1, 0, 0];

  for(i=0;i<example.length;i++) {
    // sample the next pen's states from our probability distribution
    [dx, dy, pen_down, pen_up, pen_end] = example[i];

    if (prev_pen[2] == 1) { // end of drawing.
      break;
    }

    // only draw on the paper if the pen is touching the paper
    if (prev_pen[0] == 1) {
      stroke(line_color);
      strokeWeight(line_width);
      line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
    }

    // update the absolute coordinates from the offsets
    x += dx;
    y += dy;

    // update the previous pen's state to the current one we just sampled
    prev_pen = [pen_down, pen_up, pen_end];
  }

  return [x, y]; // return final coordinates.

};

var encode_strokes = function() {
  model_state = rnn_model.zero_state();
  // encode strokes
  model_state = rnn_model.update(rnn_model.zero_input(), model_state);
  for (var i=0;i<strokes.length;i++) {
    model_state = rnn_model.update(strokes[i], model_state);
  }
};

var setup = function() {
  var rnn_model_data = JSON.parse(model_raw_data);
  rnn_model = new SketchRNN(rnn_model_data);
  rnn_model.set_pixel_factor(2.0);

  // make sure we enforce some minimum size of our demo
  screen_width = Math.max(window.innerWidth, 480);
  screen_height = Math.max(window.innerHeight, 320);

  // start drawing from somewhere in middle of the canvas
  start_x = screen_width/2.0;
  start_y = screen_height/2.0;

  // make the canvas and clear the screens
  createCanvas(screen_width, screen_height);
  frameRate(30);

  // reinitialize variables before calling p5.js setu
  line_color = color(0, 0, 220);
  predict_line_color = color(random(64, 224), random(64, 224), random(64, 224));

  // draws original strokes
  [end_x, end_y] = draw_example(strokes, start_x, start_y, line_color);

  encode_strokes();

  // copies over the model
  model_x = end_x;
  model_y = end_y;
  model_prev_pen = [0, 1, 0];

};

var draw = function() {

  var model_dx, model_dy;
  var model_pen_down, model_pen_up, model_pen_end;

  model_pdf = rnn_model.get_pdf(model_state);
  [model_dx, model_dy, model_pen_down, model_pen_up, model_pen_end] = rnn_model.sample(model_pdf, temperature);

  if (model_pen_end === 1) {
    noLoop(); // finish drawing.
  } else {

    if (model_prev_pen[0] === 1) {

      // draw line connecting prev point to current point.
      stroke(predict_line_color);
      strokeWeight(line_width);
      line(model_x, model_y, model_x+model_dx, model_y+model_dy);
    }

    model_prev_pen = [model_pen_down, model_pen_up, model_pen_end];
    model_state = rnn_model.update([model_dx, model_dy, model_pen_down, model_pen_up, model_pen_end], model_state);

    model_x += model_dx;
    model_y += model_dy;
  }
};

