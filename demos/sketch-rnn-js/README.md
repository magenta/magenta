# Sketch-RNN-JS: A Generative Model for Vector Drawings

This repo contains a JavaScript implementation for `sketch-rnn`, the recurrent neural network model described in [Teaching Machines to Draw](https://research.googleblog.com/2017/04/teaching-machines-to-draw.html) and [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477). To try some of these demos now, please read our blog post [Draw Together with a Neural Network](https://magenta.tensorflow.org/sketch_rnn_demo).

![Example Images](https://cdn.rawgit.com/tensorflow/magenta/master/magenta/models/sketch_rnn/assets/sketch_rnn_examples.svg)

*Examples of vector images produced by this generative model.*

This document is an introduction on how to use the Sketch RNN model in Javascript to generate images.  The Sketch RNN model is trained on stroke-based vector drawings.  The model is able to handle unconditional generation of vector images.

Alternatively, the model can also act as an autoencoder.  We can feed in an existing image into the model's encoder, and obtain a vector of 128 floating point numbers (the *latent vector*) that represents this image.  We can take this vector and feed it into the model's decoder, to generate a similar looking image.

For more information, please read original [model](https://magenta.tensorflow.org/sketch_rnn) description.

## Using Sketch-RNN

In the .html files, we need to include sketch\_rnn.js. Our examples are built with p5.js, so we have also included p5 libraries here too. Please see this minimal example:

```html
<html>
<head>
  <meta charset="UTF-8">
  <script language="javascript" type="text/javascript" src="lib/p5.min.js"></script>
  <script language="javascript" type="text/javascript" src="lib/p5.dom.min.js"></script>
  <script language="javascript" type="text/javascript" src="lib/numjs.js"></script>
  <script language="javascript" type="text/javascript" src="sketch_rnn.js"></script>
  <script language="javascript" type="text/javascript" src="my_sketch.js"></script>
</style>
</head>
<body>
  <div id="sketch"></div>
</body>
</html>
```

## Pre-trained models

We have provided around 100 pre-trained sketch-rnn models. We have trained each model either as the decoder model only (with a .gen.json extension), or as the full variational auto-encoder model (with a .vae.json extension). Use the vae model if you plan on using latent vectors, otherwise use the gen models.

The models are located in:

`https://storage.googleapis.com/quickdraw-models/sketchRNN/large_models/category.type.json`

where *category* is a quickdraw category such as *cat*, *dog*, *the\_mona\_lisa* etc., and *type* is either *gen* or *vae*. Some models are trained on more than one category, such as *catpig* or *crabrabbitfacepig*.

i.e.

`https://storage.googleapis.com/quickdraw-models/sketchRNN/large_models/spider.vae.json`

or

`https://storage.googleapis.com/quickdraw-models/sketchRNN/large_models/the_mona_lisa.gen.json`

Here is a list of all the models provided:

|Models   | | | | |
|---|---|---|---|---|
|alarm_clock|ambulance|angel|ant|antyoga|
|backpack|barn|basket|bear|bee|
|beeflower|bicycle|bird|book|brain|
|bridge|bulldozer|bus|butterfly|cactus|
|calendar|castle|cat|catbus|catpig|
|chair|couch|crab|crabchair|crabrabbitfacepig|
|cruise_ship|diving_board|dog|dogbunny|dolphin|
|duck|elephant|elephantpig|eye|face|
|fan|fire_hydrant|firetruck|flamingo|flower|
|floweryoga|frog|frogsofa|garden|hand|
|hedgeberry|hedgehog|helicopter|kangaroo|key|
|lantern|lighthouse|lion|lionsheep|lobster|
|map|mermaid|monapassport|monkey|mosquito|
|octopus|owl|paintbrush|palm_tree|parrot|
|passport|peas|penguin|pig|pigsheep|
|pineapple|pool|postcard|power_outlet|rabbit|
|rabbitturtle|radio|radioface|rain|rhinoceros|
|rifle|roller_coaster|sandwich|scorpion|sea_turtle|
|sheep|skull|snail|snowflake|speedboat|
|spider|squirrel|steak|stove|strawberry|
|swan|swing_set|the_mona_lisa|tiger|toothbrush|
|toothpaste|tractor|trombone|truck|whale|
|windmill|yoga|yogabicycle|everything (※)||

※ *gen* only

You should load and parse the json files using whatever method you are comfortable with, and pass the parsed blob into the constructor of the sketch\_rnn object. We will get to this later.

We found it sometimes simpler to copy the contents of the json file and place it into an inline .js code so that the demo loads the model synchronously. Some of our examples below will do this.

## Run Pre-built Examples

There are a number of proof-of-concept demos built to use the Sketch RNN model.  You can look at the corresponding code to study in detail how the model works.  To run these examples, it is recommended to use a simple local webserver, such as the http-server that can be obtained using npm, and load the local html file from the local server.  Some examples require this, since they need to dynamically load .json model files, and local static session doesn't allow for this in many browsers.

If you use the http-server, running something would be like putting `http://127.0.0.1:8080/basic_predict.html` in the address tab in Chrome.  For debugging, it is recommended you open a console tab on the side of the screen to look at the log messages.

### 1) basic\_vae.html / basic\_vae.js

This basic demo will generate an unconditional images on the web page given random latent vectors.  In addition, we demonstrate what an image looks like if we average the two latent vectors.

### 2) basic\_predict.html / basic\_predict.js

Similar to basic\_grid, this demo will keep on generating random vector images unconditionally.  Unlike basic\_vae, each point is generated per time frame (at 30 or 60 fps), while basic\_vae generates the entire image at once.  In basic, you can adjust the "temperature" variable, which controls the uncertainty of the strokes.

### 3) simple_predict.html / simple_predict.js

This demo is also generates unconditionally, attempting to finish the drawing that the user starts.
 If the user doesn't draw anything, the computer will keep on drawing stuff from scratch.

Hitting restart will clear the current human drawing and start from scratch.

In this demo, you can also select other classes, like "cat", "snail", "duck", "bus", etc.  The demo will dynamically load the json files in the models directory but cache previously loaded json models.

### 4) predict.html / predict.js

Same as the previous demo, but made to be interactive so the user can draw the beginning of a sketch on the canvas. Similar to the first [AI experiment](https://magenta.tensorflow.org/sketch-rnn-demo).

### 5 interp.html / interp.js

This demo uses the conditional generative model, and samples 2 different images (using 2 latent space vectors encoded by samples from the evaluation set).  These 2 auto-encoded images will be displayed at two sides of the screen, and the images generated in between the 2 sides will be the interpolated images based off linear-interpolation of the 128-dim latent vectors.  In the future, for better effect, spherical interpolation rather than linear can be used.

### 6) multi_vae.html / multi_vae.js

The demo is a variational autoencoder built to mimic your drawings and produce similar drawings. You are to draw a complete drawing of a specified object. After you draw a complete sketch inside the area on the left, hit the auto-encode button and the model will start drawing similar sketches inside the smaller boxes on the right. Rather than drawing a perfect duplicate copy of your drawing, the model will try to mimic your drawing instead. You can experiment drawing objects that are not the category you are supposed to draw, and see how the model interprets your drawing. For example, try to draw a cat, and have a model trained to draw crabs generate cat-like crabs.

### 7) multi_predict.html / multi_predict.js

The demo is similar to `simple_predict`. In this version, you will draw the beginning of a sketch inside the area on the left, and the model will predict the rest of the drawing inside the smaller boxes on the right. This way, you can see a variety of different endings predicted by the model. You can also choose different categories to get the model to draw different objects based on the same incomplete starting sketch. For example, you can get the model to draw things like square cats or circular trucks. You can always interrupt the model and continue working on your drawing inside the area on the left, and have the model continually predict where you left off afterwards.

### 8) simplify_lines.html / simplify_lines.js

This one does not use a machine learning model at all.  We demonstrate how  data\_tool.js is used to help us simplify lines.  When you draw something on the screen, after you release the mouse, the line you have just drawn will be automatically simplified using the RDP algorithm with an epsilon parameter of 2.0.  All models are trained to assume simplified line data with epsilon 2.0, so for best effect it is wise to convert all input data with `DataTool.simplify_lines()` function (a very efficient JS implementation of RDP), before using `DataTool.lines_to_strokes()` to convert to stroke-based dataformat for sketch\_rnn.js model to process.

## Usage of Sketch RNN model

### Pre-trained weight files

The RNN model has 2 modes: unconditional and conditional generation.  Unconditional generation means the model will just generate a random vector image from scratch and not use any latent vectors as an input.  Conditional generation mode requires a latent vector (128-dim) as an input, and whatever the model generates will be defined by those 128 numbers that can control various aspects of the image.

Whether conditional or not, all of the raw weights of these models are individually stored as .json files inside the models directory.  For example, for the 'butterfly' class, there are 2 models that come pretrained:

butterfly.gen.json - unconditional model

butterfly.vae.json - conditional model

In addition to the neural network weight matrices, there are several meta information stored in each of these files, including the version of the model, name of the class, the actual reconstruction and KL losses obtained for the evaluation set, the size of the training set used, the scale factor used to normalize the data, etc.

Some of these models are also stored for convenience as .js format, in case you just want to load a single model synchronously within the context of a pure static website demo.

### Sketch RNN

The main model is stored inside sketch\_rnn.js.  Before using the model, you need some method to import the desired .json pre-trained weight file, and parse that into a JS object first.

To create a model:

```javascript
var model_data = Parsed_JSON_file_of_pretrained_weights();
var model = new SketchRNN(model_data); // this creates your model using the pre-trained weights inside model_data
```

Currently, once you create a model, you cannot replace the weights with another JSON file, and must instead destroy this object and create another new SketchRNN object using another model_data.

To view the meta information for the pre_trained weights, just do a console.log(model.info) to dump it out.

### Scale Factors

When training the models, all the offset data has been normalized to have a standard deviation of 1.0 on the training set, after simplifying the strokes.  Neural nets work best when training on normalized data.  However, the original data recorded with the QuickDraw web app stored everything as pixels, which was scaled down so that on average the stroke offsets are ~ 1.0 length.  Thus each dataclass has its own `scale_factors` to scale down, and these numbers are usually between 60 to 120 depending on the dataset.  These scale factors are stored into `model.info.scale_factor`.  The model will assume all inputs and outputs to be in pixel space, not normalized space, and will do all the scaling for you.  You can modify these in the model using `model.set_scale_factor()`, but it is not recommended.  Rather than overwriting the `scale_factor`, modify the pixel_factor instead, as described in the next paragraph.

If using PaperJS, it is recommended that you leave everything as it is.  When using P5.JS, all the recorded data looks much bigger compared to the original app by a factor of exactly 2, and this is likely due to anti-aliasing functionality of web browsers.  Hence the extra scaling factor for the model called `pixel_factor`.  If you want to make interactive apps and receive realtime drawing data from the user, and you are using PaperJS, it is best to set do a `model.set_pixel_factor(1.0)`.  For p5.js, do a `model.set_pixel_factor(2.0)`.  For non-interactive applications, using a larger `set_pixel_factor` will reduce the size of the generated image.

### Line Data vs Stroke Data

Data collected by the original quickdraw app are stored in the below format, which is a list of list of ["x", "y"] pixel points.

```
[[["x": 123, "y": 456], ["x": 127, "y": 454], ["x": 137, "y": 450], ["x": 147, "y": 440],  ...], ...]
```

The first thing to do is to convert this format into line format, and get rid of the "x" and "y" orderings.  In the Line Data format, x always come before y:

```
Line Data: [[[123, 456], [127, 454], [137, 450], [147, 440],  ...], ...]
```

With the data\_tool.js, this Line Data format must be first simplified using `simplify_lines` or `simplify_line` (depending if it is a list of polylines or just a single polyline) first.  Afterwards, the simplified line will be fed into lines_to_strokes to convert into the Stroke Data format used by the model.

In the Stroke Data format, we assume the drawing starts at the origin, and store only the offset points from the previous location.  The format is 2 dimensional, rather than 3 dimensional as in the Line Data format:

Each row of the stroke will be 5 elements:

```
[dx, dy, p0, p1, p2]
```

`dx, dy` are the offsets in pixels from the previous point.

`p0, p1, p2` are binary values, and only one of them will be 1, the other 2 must be 0.

```
p0 = 1 means the pen stays on the paper at the next stroke.
p1 = 1 means the pen will is now above the paper after this stroke.  The next stroke will be the start of a new line.
p2 = 1 means the drawing has stopped.  Stop drawing anything!
```

The drawing will be decomposed into a list of `[dx, dy, p0, p1, p2]` strokes.

The mapping from Line Data to Stroke Data will lose the information about the starting position of the drawing, so you may want to record `LineData[0][0]` to keep this info.

## Unconditional Generation of Vector Images

### Unconditional Generation - Everything at once

Now that the preliminaries of data format and line simplification is out of the way, let's generate some vector images.

The most basic way to generate a vector image is to use an unconditional model, ie loading `ant.gen.json` into `model_data` and creating a `model = new SketchRNN(model_data)`;

To generate an entire drawing, as stroke data format:
```javascript
var example = model.generate();
```
And draw that out using your favourite method onto the canvas, or as svg's.  That's it!

There are more bells and whistles though.  You can specify a temperature parameter to specify the uncertainty and amount of variation of the image.  I recommend keeping this parameter between 0.1 to 1.0.
```javascript
var temperature = 0.40; // or can be from a DOM slider bar normalized to 0.1 -> 1.0.
var example = model.generate(temperature);

draw_example(example, 60, 100, new p.color(185, 23, 96)); // my custom method in basic_grid.js for p5.js
```
If you have written a simple `draw_example` routine like me (i.e. in basic\_predict.js), and want to center and scale the image before rendering it, there are some tools in the model to do this.

Say you have generated a cat using already using example = `model.generate(temperature)`, and want to draw that cat into a 100x100px bounding box between (10, 50) and (110, 150).  You can scale the image first, and then center it before plotting it out.
```javascript
var mod_example = model.scale_drawing(example, 100); // scales the drawing proportionally to have max size of 100px
mod_example = model.center_drawing(example)
draw_example(mod_example, 60, 100, new p.color(185, 23, 96));
```
This will draw a scaled drawing to fill the bounding box and draw it at the center.  Note that this creates a new list called mod_example to store the modified version in order to keep the original example list unmodified.

### Unconditional Generation - One Stroke at a Time

If you want to get the model to generate a stroke at a time, you can use the previous method to pre-generate the entire image, and then plot it out once every 1/60 seconds.  Alternatively, you may want to distribute the computing power and generate on the fly.  This is useful for interactive applications.

To generate a stroke at a time, let's study basic.js, a p5.js example.  Almost pseudo-code:
```javascript
var model;
var dx, dy; // offsets of the pen strokes, in pixels
var pen_down, pen_up, pen_end; // keep track of whether pen is touching paper
var x, y; // absolute coordinates on the screen of where the pen is
var prev_pen = [1, 0, 0]; // group all p0, p1, p2 together
var rnn_state; // store the hidden states of rnn's neurons
var pdf; // store all the parameters of a mixture-density distribution
var temperature = 0.65; // controls the amount of uncertainty of the model
var line_color;

var setup = function() {
  model = new SketchRNN(model_data); // assume we have a model_data
  p.createCanvas(screen_width, screen_height);
  p.frameRate(60);

  // initialize the scale factor for the model. Bigger -> large outputs
  model.set_pixel_factor(2.0);

  // initialize pen's states to zero.
  [dx, dy, pen_down, pen_up, pen_end] = model.zero_input(); // the pen's states

  // zero out the rnn's initial states
  rnn_state = model.zero_state();

  // define color of line
  line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));
};

var draw = function() {
  // see if we finished drawing
  if (prev_pen[2] == 1) {
    p.noLoop(); // stop drawing
    return;
  }

  // using the previous pen states, and hidden state, get next hidden state
  // the below line takes the most CPU power, especially for large models.
  rnn_state = model.update([dx, dy, pen_down, pen_up, pen_end], rnn_state);

  // get the parameters of the probability distribution (pdf) from hidden state
  pdf = model.get_pdf(rnn_state);

  // sample the next pen's states from our probability distribution
  [dx, dy, pen_down, pen_up, pen_end] = model.sample(pdf, temperature);

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
};
```
In the above example, using p5.js framework, the setup method is called first to initialize everything.  Afterwards, `draw()` is called 60 times a second, until `noLoop()` is called when we finish.  If you want to use the same model to draw other things again in the same session, just reinitialize the `rnn_state` like in the `setup()` function.  You should use another routine like `init()` or `restart()` to do this and not rely on the p5.js `setup()` routine.

## Variational Autoencoder - Conditional Generation of Vector Images

In this section, you will see how to use the model to encode a given vector image into a 128-dimension vector of floating point numbers (the "latent vector" Z), and also how to take a given Z (which can be either previously encoded, modified by the user, or even entirely generated), and decode it into a vector image.

To create a model, say from the cat class, you must choose between one of ant\_vae.json.

### Encoding a Vector Image into Latent Space

The encoding function, by itself, may be useful for t-sne or clustering applications.  To encode an image from the raw quickdraw data, it must first be converted to stroke data format as described earlier using DataTool object.

1. removal of "x", "y" from data, and put into list of polylines in [x, y]'s
2. simplify the line with DataTool.simplify_lines
3. convert the line to stroke format using DataTool.lines_to_strokes

After this process, say if you store the final example in a variable called example, you can encode this example to latent space using:
```javascript
var z = model.encode(example);
```
Unlike the traditional VAE paper, z is deterministic for a given example.  If we want to encode like the original VAE, and make z be a random variable, you can use an optional temperature element:
```javascript
var z = model.encode(example, encoding_temperature); // encoding_temperature between 0 and 1.
```
The 2nd method provides more artistic variation, but which is best for you depends on your application.  If you are doing clustering and prefer more certainty, then the default method may be better and easier to debug.

If you collect a group of z's, you can do PCA or t-sne or other clustering methods to analyze your data, and even use the z's for classification.  As we may upgrade the model weights in the future, each model has a versioning system stored in model.info.version, so you may want to keep track of the model version with the z's of each class if you intend to save them to use at a later point.

### Decoding a Latent Vector into a Vector Image

Assume you obtained z earlier via encoding, you can convert it back into a vector image like in the below:
```javascript
var reconstruction = model.decode(z, temperature); // temperature between 0.01 to 1.0 recommended.
```
The process of reconstruction is also a stochastic process.  This means for a given z, you can, running model.decode a few times will give you different reconstruction images.  Keeping temperature = 0.01-0.1 will give you generally very similar images and this is useful for animation applications.

For models with very low KL loss, ie < 0.50, you can even sample z from a gaussian distribution and use that z to produce a legit vector image.  To sample z from a gaussian distribution:
```javascript
var z = model.random_latent_vector();
var reconstruction = model.decode(z, temperature);
```
For some applications, you may want to bound your latent vector to a space between 0.01 to 0.99 so that everything can fit into a rectangular screen (as in interp.html/interp.js). 

# Citation

If you find this project useful for academic purposes, please cite it as:

```
@ARTICLE{sketchrnn,
  author          = {{Ha}, David and {Eck}, Douglas},
  title           = "{A Neural Representation of Sketch Drawings}",
  journal         = {ArXiv e-prints},
  archivePrefix   = "arXiv",
  eprinttype      = {arxiv},
  eprint          = {1704.03477},
  primaryClass    = "cs.NE",
  keywords        = {Computer Science - Neural and Evolutionary Computing, Computer Science - Learning, Statistics - Machine Learning},
  year            = 2017,
  month           = apr,
}
```

[arXiv]: https://arxiv.org/abs/1704.03477
[blog]: https://research.googleblog.com/2017/04/teaching-machines-to-draw.html
[dataset]: https://magenta.tensorflow.org/datasets/sketchrnn
