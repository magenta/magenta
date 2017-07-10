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
 * to interpolate between two randomly generated sketches.
 */

var sketch = function( p ) { 
  "use strict";

  var small_class_list = ['mosquito',
    'ant',
    'antyoga',
    'alarm_clock',
    'ambulance',
    'angel',
    'backpack',
    'barn',
    'basket',
    'bear',
    'bee',
    'beeflower',
    'bicycle',
    'bird',
    'book',
    'brain',
    'bridge',
    'bulldozer',
    'bus',
    'butterfly',
    'cactus',
    'calendar',
    'castle',
    'cat',
    'catbus',
    'catpig',
    'chair',
    'couch',
    'crab',
    'crabchair',
    'crabrabbitfacepig',
    'cruise_ship',
    'diving_board',
    'dog',
    'dogbunny',
    'dolphin',
    'duck',
    'elephant',
    'elephantpig',
    'eye',
    'face',
    'fan',
    'fire_hydrant',
    'firetruck',
    'flamingo',
    'flower',
    'floweryoga',
    'frog',
    'frogsofa',
    'garden',
    'hand',
    'hedgeberry',
    'hedgehog',
    'helicopter',
    'kangaroo',
    'key',
    'lantern',
    'lighthouse',
    'lion',
    'lionsheep',
    'lobster',
    'map',
    'mermaid',
    'monapassport',
    'monkey',
    'octopus',
    'owl',
    'paintbrush',
    'palm_tree',
    'parrot',
    'passport',
    'peas',
    'penguin',
    'pig',
    'pigsheep',
    'pineapple',
    'pool',
    'postcard',
    'power_outlet',
    'rabbit',
    'rabbitturtle',
    'radio',
    'radioface',
    'rain',
    'rhinoceros',
    'rifle',
    'roller_coaster',
    'sandwich',
    'scorpion',
    'sea_turtle',
    'sheep',
    'skull',
    'snail',
    'snowflake',
    'speedboat',
    'spider',
    'squirrel',
    'steak',
    'stove',
    'strawberry',
    'swan',
    'swing_set',
    'the_mona_lisa',
    'tiger',
    'toothbrush',
    'toothpaste',
    'tractor',
    'trombone',
    'truck',
    'whale',
    'windmill',
    'yoga',
    'yogabicycle'];

  var large_class_list = ['mosquito',
    'ant',
    'ambulance',
    'angel',
    'alarm_clock',
    'antyoga',
    'backpack',
    'barn',
    'basket',
    'bear',
    'bee',
    'beeflower',
    'bicycle',
    'bird',
    'book',
    'brain',
    'bridge',
    'bulldozer',
    'bus',
    'butterfly',
    'cactus',
    'calendar',
    'castle',
    'cat',
    'catbus',
    'catpig',
    'chair',
    'couch',
    'crab',
    'crabchair',
    'crabrabbitfacepig',
    'cruise_ship',
    'diving_board',
    'dog',
    'dogbunny',
    'dolphin',
    'duck',
    'elephant',
    'elephantpig',
    'eye',
    'face',
    'fan',
    'fire_hydrant',
    'firetruck',
    'flamingo',
    'flower',
    'floweryoga',
    'frog',
    'frogsofa',
    'garden',
    'hand',
    'hedgeberry',
    'hedgehog',
    'helicopter',
    'kangaroo',
    'key',
    'lantern',
    'lighthouse',
    'lion',
    'lionsheep',
    'lobster',
    'map',
    'mermaid',
    'monapassport',
    'monkey',
    'octopus',
    'owl',
    'paintbrush',
    'palm_tree',
    'parrot',
    'passport',
    'peas',
    'penguin',
    'pig',
    'pigsheep',
    'pineapple',
    'pool',
    'postcard',
    'power_outlet',
    'rabbit',
    'rabbitturtle',
    'radio',
    'radioface',
    'rain',
    'rhinoceros',
    'rifle',
    'roller_coaster',
    'sandwich',
    'scorpion',
    'sea_turtle',
    'sheep',
    'skull',
    'snail',
    'snowflake',
    'speedboat',
    'spider',
    'squirrel',
    'steak',
    'stove',
    'strawberry',
    'swan',
    'swing_set',
    'the_mona_lisa',
    'tiger',
    'toothbrush',
    'toothpaste',
    'tractor',
    'trombone',
    'truck',
    'whale',
    'windmill',
    'yoga',
    'yogabicycle'];

  var use_large_model = false;
  var class_list = small_class_list;

  if (use_large_model) {
    class_list = large_class_list;
  }

  // sketch_rnn model
  var rnn_model;
  var rnn_model_data;
  var temperature = 0.01;

  // UI
  var screen_width, screen_height, temperature_slider;
  var line_width = 1.0;
  var line_color_1, line_color_2;
  var line_color; // for rendering.

  // dom
  var model_sel;
  var text_title;
  var reset_1;
  var reset_2;
  var interpolate_button;

  var interpolate_mode = false;

  var image_spacing;

  // drawings (this can actually be used for 4-way interp, but we only use 2-way)

  var Ncol = 10;
  var Nrow = 2;

  // variables we need for this demo
  var dx, dy; // offsets of the pen strokes, in pixels
  var pen_down, pen_up, pen_end; // keep track of whether pen is touching paper
  var prev_pen = [1, 0, 0];
  var x, y; // absolute coordinates on the screen of where the pen is

  var latent_1, latent_2;
  var latent_grid = [];
  var color_grid = [];
  var drawing_grid = [];
  var interp_grid = [];

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

  };

  var redraw_screen = function() {
    p.background(255, 255, 255, 255);
    p.fill(255, 255, 255, 255);
    p.noStroke();
    p.textFont("Courier New");
    p.fill(0);
    p.textSize(12);
    p.text("temperature: "+temperature, screen_width-140, screen_height/2+image_spacing+27);
    p.stroke(1.0);
    p.strokeWeight(1.0);

    var i, c, line_color, reconstruction;
    var x, y;
    y = screen_height/2;
    for (i=0;i<Ncol;i++) {
      reconstruction = drawing_grid[i];
      if (reconstruction != null) {
        x = image_spacing*(i+0.5);
        c = color_grid[i];
        line_color = p.color(c[0], c[1], c[2]);
        draw_example(reconstruction, x, y, line_color);
      }
    }

  };

  var make_interp_grid = function(x00, x01, x10, x11) {
    // creates an interpolation grid.
    var result = [];
    var i, j;
    var d_row = 1/(Nrow-1), d_col = 1/(Ncol-1);
    var x_start, x_end;

    for(i=0;i<Nrow;i++) {
      var y = new Array(Ncol);
      x_start = x00+(x10-x00)*d_row*i;
      x_end = x01+(x11-x01)*d_row*i;
      for(j=0;j<Ncol;j++) {
        y[j] = x_start + (x_end-x_start)*d_col*j;
      }
      result.push(y);
    }

    return result;
  }

  var get_interpolation = function(obj_1, obj_2) {

    var weight_1 = make_interp_grid(1, 0, 0, 0)[0];
    var weight_2 = make_interp_grid(0, 1, 0, 0)[0];

    var len = obj_1.length;
    var result = [];
    var i, j;
    for (i=0;i<Ncol;i++) {
      result.push(new Array(len));
    }
    for (i=0;i<Ncol;i++) {
      for (j=0;j<len;j++) {
        result[i][j] = weight_1[i] * obj_1[j] + weight_2[i] * obj_2[j];
      }
    }
    return result;
  };

  var restart = function() {
    // reinitialize variables before calling p5.js setup.
    line_color_1 = [rnn_model.randi(64, 224), rnn_model.randi(64, 224), rnn_model.randi(64, 224)];
    line_color_2 = [rnn_model.randi(64, 224), rnn_model.randi(64, 224), rnn_model.randi(64, 224)];

    color_grid = get_interpolation(line_color_1, line_color_2);
    latent_grid = get_interpolation(latent_1, latent_2);

    drawing_grid = [];
    for (var i=0;i<Ncol;i++) {
      drawing_grid.push(null);
    }

  };

  var restart_2 = function() {
    // reinitialize variables before calling p5.js setup.
    line_color_2 = [rnn_model.randi(64, 224), rnn_model.randi(64, 224), rnn_model.randi(64, 224)];

    color_grid = get_interpolation(line_color_1, line_color_2);
    latent_grid = get_interpolation(latent_1, latent_2);

    for (var i=1;i<Ncol;i++) {
      drawing_grid[i] = null;
    }

  };

  var restart_1 = function() {
    // reinitialize variables before calling p5.js setup.
    line_color_1 = [rnn_model.randi(64, 224), rnn_model.randi(64, 224), rnn_model.randi(64, 224)];

    color_grid = get_interpolation(line_color_1, line_color_2);
    latent_grid = get_interpolation(latent_1, latent_2);

    for (var i=0;i<Ncol-1;i++) {
      drawing_grid[i] = null;
    }

  };

  p.setup = function() {

    // make sure we enforce some minimum size of our demo
    screen_width = Math.max(window.innerWidth, 480);
    screen_height = Math.max(window.innerHeight, 320);

    // figure out size of each drawing that fits in the tile
    image_spacing = screen_width/Ncol;

    // declare sketch_rnn model
    ModelImporter.set_init_model(model_raw_data);
    if (use_large_model) {
      ModelImporter.set_model_url("https://storage.googleapis.com/quickdraw-models/sketchRNN/large_models/");      
    }
    rnn_model_data = ModelImporter.get_model_data();
    rnn_model = new SketchRNN(rnn_model_data);

    text_title = p.createP("sketch-rnn interpolation.");
    text_title.style("font-family", "monospace");
    text_title.style("font-size", "16");
    text_title.style("color", "#3393d1"); // ff990a
    text_title.position(10, screen_height/2-image_spacing-27-20-40);

    // model selection
    model_sel = p.createSelect();
    model_sel.position(10, screen_height/2+image_spacing+27);
    for (var i=0;i<class_list.length;i++) {
      model_sel.option(class_list[i]);
    }
    model_sel.changed(model_sel_event);

    // temp
    temperature_slider = p.createSlider(1, 100, temperature*100);
    temperature_slider.position(145, screen_height/2+image_spacing+27);
    temperature_slider.style('width', screen_width-20-145+'px');
    temperature_slider.changed(temperature_slider_event);

    // dom
    reset_1 = p.createButton('randomize #1');
    reset_1.position(5, screen_height/2-image_spacing-27-20);
    reset_1.mousePressed(reset_1_event); // attach button listener

    // dom
    reset_2 = p.createButton('randomize #2');
    reset_2.position(screen_width-101, screen_height/2-image_spacing-27-20);
    reset_2.mousePressed(reset_2_event); // attach button listener

    // interp button
    interpolate_button = p.createButton('Interpolate!');
    interpolate_button.position(screen_width/2-40, screen_height/2-image_spacing-27-20);
    interpolate_button.mousePressed(interpolate_button_event); // attach button listener

    // make the canvas and clear the screens
    p.createCanvas(screen_width, screen_height);
    p.frameRate(30);

    // calculate random latent vectors.
    latent_1 = rnn_model.random_latent_vector();
    latent_2 = rnn_model.random_latent_vector();
    restart();

    redraw_screen();

  };

  var process_drawing = function(n) {
    if (drawing_grid[n] === null) { // only do something if it is not null
      var z = latent_grid[n];
      var reconstruction = rnn_model.decode(z, temperature);
      reconstruction = rnn_model.scale_drawing(reconstruction, image_spacing*0.70);
      reconstruction = rnn_model.center_drawing(reconstruction);
      drawing_grid[n] = reconstruction;
      redraw_screen();
    }
  };

  p.draw = function() {

    if (drawing_grid[0] === null) {
      process_drawing(0);
      return;
    }
    if (drawing_grid[Ncol-1] === null) {
      process_drawing(Ncol-1);
      return;
    }
    if (interpolate_mode) {
      for (var i=1;i<Ncol-1;i++) {
        if (drawing_grid[i] === null) {
          process_drawing(i);
          return;
        }
      }
    }

    if (interpolate_mode) {
      interpolate_mode = false;
    }

  };

  var temperature_slider_event = function() {
    temperature = temperature_slider.value()/100;
    redraw_screen();
    console.log("set temperature to "+temperature);
    restart();
  };

  var model_sel_event = function() {
    var c = model_sel.value();
    var model_mode = "vae";
    console.log("user wants to change to model "+c);
    var call_back = function(new_model) {
      rnn_model = new_model;
      restart();
    }
    ModelImporter.change_model(rnn_model, c, model_mode, call_back);
  };

  var reset_1_event = function() {
    latent_1 = rnn_model.random_latent_vector();
    restart_1();
  };

  var reset_2_event = function() {
    latent_2 = rnn_model.random_latent_vector();
    restart_2();
  };

  var interpolate_button_event = function() {
    interpolate_mode = true;
  };

};
var custom_p5 = new p5(sketch, 'sketch');
