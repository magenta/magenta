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
 * to finish the user's incomplete drawing, and loop through different
 * endings automatically.
 */
var sketch = function( p ) { 
  "use strict";

  var small_class_list = ['ant',
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
    'mosquito',
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

  var large_class_list = ['ant',
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
    'everything',
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
    'mosquito',
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

  var use_large_models = true;

  var class_list = small_class_list;

  if (use_large_models) {
    class_list = large_class_list;
  }

  // sketch_rnn model
  var model;
  var model_data;
  var temperature = 0.25;
  var min_sequence_length = 5;

  var model_pdf; // store all the parameters of a mixture-density distribution
  var model_state, model_state_orig;
  var model_prev_pen;
  var model_dx, model_dy;
  var model_pen_down, model_pen_up, model_pen_end;
  var model_x, model_y;
  var model_is_active;

  // variables for the sketch input interface.
  var pen;
  var prev_pen;
  var x, y; // absolute coordinates on the screen of where the pen is
  var start_x, start_y;
  var has_started; // set to true after user starts writing.
  var just_finished_line;
  var epsilon = 2.0; // to ignore data from user's pen staying in one spot.
  var raw_lines;
  var current_raw_line;
  var strokes;
  var line_color, predict_line_color;

  // UI
  var screen_width, screen_height, temperature_slider;
  var line_width = 2.0;
  var screen_scale_factor = 3.0;

  // dom
  var reset_button, model_sel, random_model_button;
  var text_title, text_temperature;

  var title_text = "sketch-rnn predictor.";

  var set_title_text = function(new_text) {
    title_text = new_text.split('_').join(' ');
    text_title.html(title_text);
    text_title.position(screen_width/2-12*title_text.length/2+10, 0);
  };

  var update_temperature_text = function() {
    var the_color="rgba("+Math.round(255*temperature)+",0,"+255+",1)";
    text_temperature.style("color", the_color); // ff990a
    text_temperature.html(""+Math.round(temperature*100));
  };

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

  };

  var init = function() {

    // model
    ModelImporter.set_init_model(model_raw_data);
    if (use_large_models) {
      ModelImporter.set_model_url("https://storage.googleapis.com/quickdraw-models/sketchRNN/large_models/");      
    }
    model_data = ModelImporter.get_model_data();
    model = new SketchRNN(model_data);
    model.set_pixel_factor(screen_scale_factor);

    screen_width = p.windowWidth; //window.innerWidth
    screen_height = p.windowHeight; //window.innerHeight

    // dom

    reset_button = p.createButton('clear drawing');
    reset_button.position(10, screen_height-27-27);
    reset_button.mousePressed(reset_button_event); // attach button listener

    // random model buttom
    random_model_button = p.createButton('random');
    random_model_button.position(117, screen_height-27-27);
    random_model_button.mousePressed(random_model_button_event); // attach button listener

    // model selection
    model_sel = p.createSelect();
    model_sel.position(195, screen_height-27-27);
    for (var i=0;i<class_list.length;i++) {
      model_sel.option(class_list[i]);
    }
    model_sel.changed(model_sel_event);

    // temp
    temperature_slider = p.createSlider(1, 100, temperature*100);
    temperature_slider.position(0*screen_width/2-10*0+10, screen_height-27);
    temperature_slider.style('width', screen_width/1-25+'px');
    temperature_slider.changed(temperature_slider_event);

    // title
    text_title = p.createP(title_text);
    text_title.style("font-family", "Courier New");
    text_title.style("font-size", "20");
    text_title.style("color", "#3393d1"); // ff990a
    set_title_text(title_text);

    // temperature text
    text_temperature = p.createP();
    text_temperature.style("font-family", "Courier New");
    text_temperature.style("font-size", "16");
    text_temperature.position(screen_width-40, screen_height-64);
    update_temperature_text(title_text);

  };

  var encode_strokes = function(sequence) {

    model_state_orig = model.zero_state();

    if (sequence.length <= min_sequence_length) {
      return;
    }

    // encode sequence
    model_state_orig = model.update(model.zero_input(), model_state_orig);
    for (var i=0;i<sequence.length-1;i++) {
      model_state_orig = model.update(sequence[i], model_state_orig);
    }

    restart_model(sequence);

    model_is_active = true;

  }

  var restart_model = function(sequence) {

    model_state = model.copy_state(model_state_orig); // bounded

    var idx = raw_lines.length-1;
    var last_point = raw_lines[idx][raw_lines[idx].length-1];
    var last_x = last_point[0];
    var last_y = last_point[1];

    // individual models:
    var sx = last_x;
    var sy = last_y;

    var dx, dy, pen_down, pen_up, pen_end;
    var s = sequence[sequence.length-1];

    model_x = sx;
    model_y = sy;

    dx = s[0];
    dy = s[1];
    pen_down = s[2];
    pen_up = s[3];
    pen_end = s[4];

    model_dx = dx;
    model_dy = dy;
    model_prev_pen = [pen_down, pen_up, pen_end];

  }

  var restart = function() {

    // reinitialize variables before calling p5.js setup.
    line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));
    predict_line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));

    // make sure we enforce some minimum size of our demo
    screen_width = Math.max(window.innerWidth, 480);
    screen_height = Math.max(window.innerHeight, 320);

    // variables for the sketch input interface.
    pen = 0;
    prev_pen = 1;
    has_started = false; // set to true after user starts writing.
    just_finished_line = false;
    raw_lines = [];
    current_raw_line = [];
    strokes = [];
    // start drawing from somewhere in middle of the canvas
    x = screen_width/2.0;
    y = screen_height/2.0;
    start_x = x;
    start_y = y;
    has_started = false;

    model_x = x;
    model_y = y;
    model_prev_pen = [0, 1, 0];
    model_is_active = false;

  };

  var clear_screen = function() {
    p.background(255, 255, 255, 255);
    p.fill(255, 255, 255, 255);
  };

  p.setup = function() {
    init();
    restart();
    p.createCanvas(screen_width, screen_height);
    p.frameRate(60);
    clear_screen();
    console.log('ready.');
  };

  // tracking mouse  touchpad
  var tracking = {
    down: false,
    x: 0,
    y: 0
  };

  p.draw = function() {
    deviceEvent();
    // record pen drawing from user:
    if (tracking.down && (tracking.x > 0) && tracking.y < (screen_height-60)) { // pen is touching the paper
      if (has_started == false) { // first time anything is written
        has_started = true;
        x = tracking.x;
        y = tracking.y;
        start_x = x;
        start_y = y;
        pen = 0;
      }
      var dx0 = tracking.x-x; // candidate for dx
      var dy0 = tracking.y-y; // candidate for dy
      if (dx0*dx0+dy0*dy0 > epsilon*epsilon) { // only if pen is not in same area
        var dx = dx0;
        var dy = dy0;
        pen = 0;

        if (prev_pen == 0) {
          p.stroke(line_color);
          p.strokeWeight(line_width); // nice thick line
          p.line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
        }

        // update the absolute coordinates from the offsets
        x += dx;
        y += dy;

        // update raw_lines
        current_raw_line.push([x, y]);
        just_finished_line = true;

        // using the previous pen states, and hidden state, get next hidden state 
        // update_rnn_state();
      }
    } else { // pen is above the paper
      pen = 1;
      if (just_finished_line) {
        var current_raw_line_simple = DataTool.simplify_line(current_raw_line);
        var idx, last_point, last_x, last_y;

        if (current_raw_line_simple.length > 1) {
          if (raw_lines.length === 0) {
            last_x = start_x;
            last_y = start_y;
          } else {
            idx = raw_lines.length-1;
            last_point = raw_lines[idx][raw_lines[idx].length-1];
            last_x = last_point[0];
            last_y = last_point[1];
          }
          var stroke = DataTool.line_to_stroke(current_raw_line_simple, [last_x, last_y]);
          raw_lines.push(current_raw_line_simple);
          strokes = strokes.concat(stroke);

          // initialize rnn:
          encode_strokes(strokes);

          // redraw simplified strokes
          clear_screen();
          draw_example(strokes, start_x, start_y, line_color);

          /*
          p.stroke(line_color);
          p.strokeWeight(2.0);
          p.ellipse(x, y, 5, 5); // draw line connecting prev point to current point.
          */

        } else {
          if (raw_lines.length === 0) {
            has_started = false;
          }
        }

        current_raw_line = [];
        just_finished_line = false;
      }

      // have machine take over the drawing here:
      if (model_is_active) {

        model_pen_down = model_prev_pen[0];
        model_pen_up = model_prev_pen[1];
        model_pen_end = model_prev_pen[2];

        model_state = model.update([model_dx, model_dy, model_pen_down, model_pen_up, model_pen_end], model_state);
        model_pdf = model.get_pdf(model_state);
        [model_dx, model_dy, model_pen_down, model_pen_up, model_pen_end] = model.sample(model_pdf, temperature);

        if (model_pen_end === 1) {
          restart_model(strokes);
          //model_pen_end = 0;
          //model_pen_down = 1;
          //model_pen_up = 1;
          predict_line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));
          clear_screen();
          draw_example(strokes, start_x, start_y, line_color);
        } else {

          if (model_prev_pen[0] === 1) {

            // draw line connecting prev point to current point.
            p.stroke(predict_line_color);
            p.strokeWeight(line_width);
            p.line(model_x, model_y, model_x+model_dx, model_y+model_dy);
          }

          model_prev_pen = [model_pen_down, model_pen_up, model_pen_end];

          model_x += model_dx;
          model_y += model_dy;
        }
      }

    } 

    prev_pen = pen;
  };

  var model_sel_event = function() {
    var c = model_sel.value();
    var model_mode = "gen";
    console.log("user wants to change to model "+c);
    var call_back = function(new_model) {
      model = new_model;
      model.set_pixel_factor(screen_scale_factor);
      encode_strokes(strokes);
      clear_screen();
      draw_example(strokes, start_x, start_y, line_color);
      set_title_text('draw '+model.info.name+'.');
    }
    set_title_text('loading '+c+' model...');
    ModelImporter.change_model(model, c, model_mode, call_back);
  };

  var random_model_button_event = function() {
    var item = class_list[Math.floor(Math.random()*class_list.length)];
    model_sel.value(item);
    model_sel_event();
  };

  var reset_button_event = function() {
    restart();
    clear_screen();
  };

  var temperature_slider_event = function() {
    temperature = temperature_slider.value()/100;
    clear_screen();
    draw_example(strokes, start_x, start_y, line_color);
    update_temperature_text();
  };

  var deviceReleased = function() {
    "use strict";
    tracking.down = false;
  }

  var devicePressed = function(x, y) {
    if (y < (screen_height-60)) {
      tracking.x = x;
      tracking.y = y;
      if (!tracking.down) {
        tracking.down = true;
      }
    }
  };

  var deviceEvent = function() {
    if (p.mouseIsPressed) {
      devicePressed(p.mouseX, p.mouseY);
    } else {
      deviceReleased();
    }
  }

};
var custom_p5 = new p5(sketch, 'sketch');
