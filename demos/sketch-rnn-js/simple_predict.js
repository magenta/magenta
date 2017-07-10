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
 * to finish a fixed incomplete drawings, and loop through multiple
 * endings automatically.
 */
var sketch = function( p ) { 
  "use strict";

  var class_list = ['bird',
    'ant',
    'angel',
    'bee',
    'bicycle',
    'flamingo',
    'flower',
    'mosquito',
    'owl',
    'spider',
    'yoga'];

  var strokes=[[-4,0,1,0,0],[-15,9,1,0,0],[-10,17,1,0,0],[-1,28,1,0,0],[14,13,1,0,0],[12,4,1,0,0],[22,1,1,0,0],[14,-11,1,0,0],[5,-12,1,0,0],[2,-19,1,0,0],[-12,-23,1,0,0],[-13,-7,1,0,0],[-14,-1,0,1,0]];

  // sketch_rnn model
  var model;
  var model_data;
  var temperature = 0.25;
  var min_sequence_length = 5;
  var screen_scale_factor = 5.0;

  var model_pdf; // store all the parameters of a mixture-density distribution
  var model_state, model_state_orig;
  var model_prev_pen;
  var model_x, model_y;

  // variables for the sketch input interface.
  var start_x, start_y;
  var end_x, end_y;

  // UI
  var screen_width, screen_height, temperature_slider;
  var line_width = 2.0;
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
        p.stroke(line_color);
        p.strokeWeight(line_width);
        p.line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
      }

      // update the absolute coordinates from the offsets
      x += dx;
      y += dy;

      // update the previous pen's state to the current one we just sampled
      prev_pen = [pen_down, pen_up, pen_end];
    }

    return [x, y]; // return final coordinates.

  };

  var clear_screen = function() {
    p.background(255, 255, 255, 255);
    p.fill(255, 255, 255, 255);
    p.noStroke();
    p.textFont("Courier New");
    p.fill(0);
    p.textSize(12);
    p.text("temperature: "+temperature, screen_width-130, screen_height-35);
    p.stroke(1.0);
    p.strokeWeight(1.0);
  };

  var restart = function() {
    // reinitialize variables before calling p5.js setup.
    line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));
    predict_line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));

    // draws original strokes
    clear_screen();
    [end_x, end_y] = draw_example(strokes, start_x, start_y, line_color);

    // copies over the model
    model_state = model.copy_state(model_state_orig);
    model_x = end_x;
    model_y = end_y;
    model_prev_pen = [0, 1, 0];
  };

  var encode_strokes = function() {
    model_state_orig = model.zero_state();
    // encode strokes
    model_state_orig = model.update(model.zero_input(), model_state_orig);
    for (var i=0;i<strokes.length;i++) {
      model_state_orig = model.update(strokes[i], model_state_orig);
    }
  };

  p.setup = function() {

    // make sure we enforce some minimum size of our demo
    screen_width = Math.max(window.innerWidth, 480);
    screen_height = Math.max(window.innerHeight, 320);

    // start drawing from somewhere in middle of the canvas
    start_x = screen_width/2.0;
    start_y = screen_height/3.0;

    // declare sketch_rnn model
    ModelImporter.set_init_model(model_raw_data);
    model_data = ModelImporter.get_model_data();
    model = new SketchRNN(model_data);
    model.set_pixel_factor(screen_scale_factor);

    // model selection
    model_sel = p.createSelect();
    model_sel.position(10, screen_height-27);
    for (var i=0;i<class_list.length;i++) {
      model_sel.option(class_list[i]);
    }
    model_sel.changed(model_sel_event);

    // temp
    temperature_slider = p.createSlider(1, 100, temperature*100);
    temperature_slider.position(95, screen_height-27);
    temperature_slider.style('width', screen_width-20-95+'px');
    temperature_slider.changed(temperature_slider_event);

    // make the canvas and clear the screens
    p.createCanvas(screen_width, screen_height);
    p.frameRate(60);

    encode_strokes();
    restart();

  };

  p.draw = function() {

    var model_dx, model_dy;
    var model_pen_down, model_pen_up, model_pen_end;

    model_pdf = model.get_pdf(model_state);
    [model_dx, model_dy, model_pen_down, model_pen_up, model_pen_end] = model.sample(model_pdf, temperature);

    if (model_pen_end === 1) {
      restart();
    } else {

      if (model_prev_pen[0] === 1) {

        // draw line connecting prev point to current point.
        p.stroke(predict_line_color);
        p.strokeWeight(line_width);
        p.line(model_x, model_y, model_x+model_dx, model_y+model_dy);
      }

      model_prev_pen = [model_pen_down, model_pen_up, model_pen_end];
      model_state = model.update([model_dx, model_dy, model_pen_down, model_pen_up, model_pen_end], model_state);

      model_x += model_dx;
      model_y += model_dy;
    }
  };

  var temperature_slider_event = function() {
    temperature = temperature_slider.value()/100;
    clear_screen();
    draw_example(strokes, start_x, start_y, line_color);
    console.log("set temperature to "+temperature);
  };

  var model_sel_event = function() {
    var c = model_sel.value();
    var model_mode = "gen";
    console.log("user wants to change to model "+c);
    var call_back = function(new_model) {
      model = new_model;
      model.set_pixel_factor(screen_scale_factor);
      encode_strokes();
      restart();
    }
    ModelImporter.change_model(model, c, model_mode, call_back);
  };

};
var custom_p5 = new p5(sketch, 'sketch');
