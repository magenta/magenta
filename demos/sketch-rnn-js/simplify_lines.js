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
 * @fileoverview Basic p5.js sketch.
 * no machine learning here. just draw something on the screen,
 * simplify it, and print out points.
 */

var sketch = function( p ) { 
  "use strict";

  // variables we need for this demo
  var dx, dy; // offsets of the pen strokes, in pixels
  var pen = 0;
  var prev_pen = 1;
  var x, y; // absolute coordinates on the screen of where the pen is
  var start_x, start_y;

  var has_started = false; // set to true after user starts writing.
  var just_finished_line = false;
  var epsilon = 2.0; // to ignore data from user's pen staying in one spot.

  var screen_width, screen_height; // stores the browser's dimensions

  var raw_lines = [];
  var current_raw_line = [];
  var current_raw_line_simple;
  var strokes = [];
  var stroke;

  var last_point, idx;

  var line_color;

  var draw_example = function(example, start_x, start_y, line_color) {
    var i;
    var x=start_x, y=start_y;
    var x, y;
    var pen_down, pen_up, pen_end;
    var prev_pen = [1, 0, 0];

    console.log("input_strokes="+JSON.stringify(example));

    for(i=0;i<example.length;i++) {
      // sample the next pen's states from our probability distribution
      [dx, dy, pen_down, pen_up, pen_end] = example[i];

      if (prev_pen[2] == 1) { // end of drawing.
        break;
      }

      // only draw on the paper if the pen is touching the paper
      if (prev_pen[0] == 1) {
        p.stroke(line_color);
        p.strokeWeight(2.0);
        p.line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
      }

      // update the absolute coordinates from the offsets
      x += dx;
      y += dy;

      // update the previous pen's state to the current one we just sampled
      prev_pen = [pen_down, pen_up, pen_end];
    }

  };

  var restart = function() {
    // reinitialize variables before calling p5.js setup.

    line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));

    // make sure we enforce some minimum size of our demo
    screen_width = Math.max(window.innerWidth, 480);
    screen_height = Math.max(window.innerHeight, 320);

    // start drawing from somewhere in middle of the canvas
    x = screen_width/2.0;
    y = screen_height/2.0;

    has_started = false;

  };

  p.setup = function() {
    restart(); // initialize variables for this demo
    p.createCanvas(screen_width, screen_height);
    p.frameRate(60);
    p.background(255, 255, 255, 255);
    p.fill(255, 255, 255, 255);
  };

  p.draw = function() {
    // record pen drawing from user:
    if (p.mouseIsPressed) { // pen is touching the paper
      if (has_started == false) { // first time anything is written
        has_started = true;
        x = p.mouseX;
        y = p.mouseY;
        start_x = x;
        start_y = y;
        pen = 0;
        /*
        p.stroke(line_color);
        p.strokeWeight(2.0);
        p.ellipse(x, y, 5, 5); // draw line connecting prev point to current point.
        */
      }
      var dx0 = p.mouseX-x; // candidate for dx
      var dy0 = p.mouseY-y; // candidate for dy
      if (dx0*dx0+dy0*dy0 > epsilon*epsilon) { // only if pen is not in same area
        dx = dx0;
        dy = dy0;
        pen = 0;

        if (prev_pen == 0) {
          p.stroke(line_color);
          p.strokeWeight(2.0); // nice thick line
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
        current_raw_line_simple = DataTool.simplify_line(current_raw_line);

        if (current_raw_line_simple.length > 1) {
          if (raw_lines.length === 0) {
            last_point = [start_x, start_y];
          } else {
            idx = raw_lines.length-1;
            last_point = raw_lines[idx][raw_lines[idx].length-1];
          }

          stroke = DataTool.line_to_stroke(current_raw_line_simple, last_point);
          raw_lines.push(current_raw_line_simple);
          strokes = strokes.concat(stroke);
          p.background(255, 255, 255, 255);
          p.fill(255, 255, 255, 255);
          draw_example(strokes, start_x, start_y, line_color);
        } else {
          if (raw_lines.length === 0) {
            has_started = false;
          }
        }

        current_raw_line = [];
        just_finished_line = false;
      }
    }

    prev_pen = pen;
  };

};
var custom_p5 = new p5(sketch, 'sketch');
