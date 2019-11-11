# Magenta MIDI Interface

This interface allows you to connect to a model
[generator](/magenta/models/README.md#generators) via a MIDI controller
and synthesizer. These can be either "hard" or "soft" components.

Note that you can only interface with a trained models that have a
[SequenceGenerator](/magenta/models/shared/sequence_generator.py)
 defined for them.

<p align="center">
  <img src="midi.png" alt="Sequence Diagram for the MIDI interface"/>
</p>

## Example Demo

The simplest way to try this interface is using the
[AI Jam demo](https://github.com/tensorflow/magenta-demos/tree/master/ai-jam-js). The instructions below provide a more basic
customizable interaction that is more difficult to set up.

## Installing Dependencies

Before using the interface, you will need to install some
dependencies. We have provided instructions for both Macintosh OS X
and Ubuntu Linux.

For users of Macintosh OS X, the instructions below assume that you
have installed [Homebrew](http://brew.sh).

First, [install Magenta](/README.md). The rest of this document assumes you have
installed the Magenta pip package. Before continuing, make sure your `magenta`
conda environment is active:

```bash
source activate magenta
```

### Install QjackCtl (Ubuntu Only)

[QjackCtl](http://qjackctl.sourceforge.net/) is a tool that provides a graphical
interface for the JACK hub on Ubuntu to allow you to easily route signals
between MIDI components. You can install it using `sudo apt-get install
qjackctl`.

### Connect/Install MIDI Controller

If you are using a hardware controller, attach it to the machine. If you do not
have one, you can install a software controller such as
[VMPK](http://vmpk.sourceforge.net/) by doing the following.

**Ubuntu:** Use the command `sudo apt-get install vmpk`.<br />
**Mac:** Download and install from the
[VMPK website](http://vmpk.sourceforge.net/#Download).

### Connect/Install MIDI Synthesizer

If you are using a hardware synthesizer, attach it to the machine. If you do not
have one, you can install a software synthesizer such as [FluidSynth]
(http://www.fluidsynth.org) using the following commands:

**Ubuntu:** `sudo apt-get install fluidsynth`<br />
**Mac:** `brew install fluidsynth`

If using FluidSynth, you will also want to install a decent soundfont. You can
install one by doing the following:

**Ubuntu:** Use the command `sudo apt-get install fluid-soundfont-gm`.<br />
**Mac:** Download the soundfont from
http://www.musescore.org/download/fluid-soundfont.tar.gz and unpack the SF2
file.

## Set Up

### Ubuntu

Launch `qjackctl`. You'll probably want to do it in its own screen/tab
since it will print status messages to the terminal. Once the GUI
appears, click the "Start" button.

If using a software controller, you can launch it in the background or in its
own screen/tab. Use `vmpk` to launch VMPK.

If using a software synth, you can launch it in the background or in its own
screen/tab. Launch FluidSynth with the recommended soundfont installed above
using:

```bash
fluidsynth /usr/share/sounds/sf2/FluidR3_GM.sf2
```

In the QjackCtl GUI, click the "Connect" button. In the "Audio" tab, select your
synthesizer from the list on the left (e.g., "fluidsynth") and select "system"
from the list on the right. Then click the "Connect" button at the bottom.

### Mac

If using a software controller (e.g., VMPK), launch it.

If using a software synth, launch it. Launch FluidSynth with the
recommended soundfont downloaded above using:

```bash
fluidsynth /path/to/sf2
```

## Launching the Interface

After completing the installation and set up steps above have the interface list
the available MIDI ports:

```bash
magenta_midi --list_ports
```

You should see a list of available input and output ports, including both the
controller (e.g., "VMPK Output") and synthesizer (e.g., "FluidSynth virtual
port"). Set the environment variables based on the ports you want to use. For
example:

```bash
CONTROLLER_PORT="VMPK Output"
SYNTH_PORT="FluidSynth virtual port 1"
```

To use the midi interface, you must supply one or more trained model bundles
(.mag files). You can either download them from the links on our model pages
(e.g., [Melody RNN](/magenta/models/melody_rnn/README.md)) or create bundle
files from your training checkpoints using the instructions on the model page.
Once you're picked out the bundle files you wish to use, set the magenta_midi --help
environment
variable with a comma-separated list of paths to to the bundles. For example:

```bash
BUNDLE_PATHS=/path/to/bundle1.mag,/path/to/bundle2.mag
```

In summary, you should first define these variables:

```bash
CONTROLLER_PORT=<controller midi port name>
SYNTH_PORT=<synth midi port name>
BUNDLE_PATHS=<comma-separated paths to bundle files>
```

You may now start the interface with this command:

```bash
magenta_midi \
  --input_ports=${CONTROLLER_PORT} \
  --output_ports=${SYNTH_PORT} \
  --bundle_files=${BUNDLE_PATHS}
```

There are many other options you can set to customize your interaction. To see
a full list, you can enter:

```bash
magenta_midi --help
```

## Assigning Control Signals
You can assign control change numbers to different "knobs" for controlling the
interface in two ways.

* Assign the values on the command line using the appropriate flags (e.g.,
`--temperature_control_number=1`).
* Assign the values after startup by dynamically associating control changes
from your MIDI controller with different control signals. You can enter the UI
for doing this assignment by including the `--learn_controls` flag on the
command-line at launch.


## Using the "Call and Response" Interaction

"Call and response" is a type of interaction where one participant (you) produce
a "call" phrase and the other participant (Magenta) produces a "response" phrase
based upon that "call".

When you start the interface, "call" phrase capture will begin immediately. You
will hear a metronome ticking and the keys will now produce sounds through your
audio output.

When you would like to hear a response, you should stop playing and a wait a
bar, at which point the response will be played. Once the response completes,
call phrase capture will begin again, and the process repeats.

If you used the `--end_call_control_number` flag, you can signal with that
control number and a value of 127 to end the call phrase instead of waiting for
a bar of silence. At the end of the current bar, a generated response will be
played that is the same length as your call phrase. After the response
completes, call phrase capture will begin again, and the process repeats.

Assuming you're using the
[Attention RNN](/magenta/models/melody_rnn/README.md#configurations) bundle file
and are using VPMK and FluidSynth, your command might look like this:

```bash
magenta_midi \
  --input_ports="VMPK Output" \
  --output_ports="FluidSynth virtual port" \
  --bundle_files=/tmp/attention_rnn.mag
```
