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
 * to generate random sketchs from a random latent vector.
 *
 * We generate 2 random sketches (in blue and green).
 *
 * We also take the average of the 2 vectors, and generate
 * this interpolate image (in yellow).
 */

var rnn_model;
var rnn_model_data;
var screen_width;
var screen_height;

var draw_example = function(example, start_x, start_y, line_color) {
  var i;
  var x=start_x, y=start_y;
  var dx, dy;
  var pen_down, pen_up, pen_end;
  var prev_pen = [0, 1, 0];

  for(i=0;i<example.length;i++) {
    // sample the next pen's states from our probability distribution
    [dx, dy, pen_down, pen_up, pen_end] = example[i];

    if (prev_pen[2] == 1) { // end of drawing.
      break;
    }

    // only draw on the paper if the pen is touching the paper
    if (prev_pen[0] == 1) {
      stroke(line_color);
      strokeWeight(1.0);
      line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
    }

    // update the absolute coordinates from the offsets
    x += dx;
    y += dy;

    // update the previous pen's state to the current one we just sampled
    prev_pen = [pen_down, pen_up, pen_end];
  }

};

var setup = function() {

  var drawing, i, temperature, x_position;

  // make sure we enforce some minimum size of our demo
  screen_width = Math.max(window.innerWidth, 480);
  screen_height = Math.max(window.innerHeight, 320);

  createCanvas(screen_width, screen_height, SVG);

  // declare sketch_rnn model
  var rnn_model_data = JSON.parse(model_raw_data);
  rnn_model = new SketchRNN(rnn_model_data);

  // temperature labels
  noStroke();
  textFont("Courier New");
  fill(0);
  textSize(20);
  for(i=0;i<5;i++) {
    x_position = 120*(i+1)-25;
    text("T="+(i/10), x_position, 20);
  }

  function generate(z, y_start, c) {
    for(i=0;i<5;i++) {
      temperature = 0.1*(1*i+0);
      x_position = 120*(i+1);
      drawing = rnn_model.decode(z, temperature);
      drawing = rnn_model.scale_drawing(drawing, 90);
      drawing = rnn_model.center_drawing(drawing);
      draw_example(drawing, x_position, y_start, c);
    }
  }

  // create a random drawing.
  var z_0 = rnn_model.random_latent_vector();
  generate(z_0, 120, color(220, 0, 0));

  // create a random drawing.
  var z_1 = rnn_model.random_latent_vector();
  generate(z_1, 270, color(0, 0, 220));

  // create a random drawing.
  var z_2 = new Array(128);
  for (var i=0;i<128;i++) {
    z_2[i] = 0.5*z_0[i] + 0.1*z_1[i];
  }
  generate(z_2, 420, color(220, 0, 220));

};
