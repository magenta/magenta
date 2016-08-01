# Magenta MIDI Interface

This interface allows you to connect to the MelodyGenerator server via a MIDI
controller and synthesizer. These can be either "hard" or "soft" components.

The instructions below are for Ubuntu.

## Installing Dependencies

### Install PortMidi

The demo code uses a python library called [mido](http://mido.readthedocs.io) to
interface your computer's MIDI hub. For it to work, you need to separately
install a backend library it can use to connect to your system. The easiest to
install is PortMidi, which can be done with the command `sudo apt-get install
libportmidi-dev`.

### Install QjackCtl

[QjackCtl](http://qjackctl.sourceforge.net/) is a tool that provides a graphical
interface for the JACK hub on your machine to allow you to easily route signals
between MIDI components. You can install it using `sudo apt-get install
qjackctl`.

### Connect/Install MIDI Controller

If you are using a hardware controller, attach it to the machine. If you do not
have one, you can install a software controller such as [vkeybd]
(http://ccrma.stanford.edu/planetccrma/man/man1/vkeybd.1.html) using `sudo
apt-get install vkeybd`.

### Connect/Install MIDI Synthesizer

If you are using a hardware synthesizer, attach it to the machine. If you do not
have one, you can install a software synthesizer such as [FluidSynth]
(http://www.fluidsynth.org) using `sudo apt-get install fluidsynth`.

If using FluidSynth, you will also want to install a decent soundfont. You can
install one using `sudo apt-get install fluid-soundfont-gm`.

## Running Interface

Once you have installed the above dependencies, you can start by launching
`qjackctl`. You'll probably want to do it in its own screen/tab since it will
print status messages to the terminal. Once the GUI appears, click the "Start"
button.

If using a software controller, you can launch it in the background or in its
own screen/tab. Use `vkeybd` to launch vkeybd.

If using a software synth, you can launch it in the background or in its own
screen/tab. Launch FluidSynth with the recommended soundfont installed above
using:

```bash
$ fluidsynth /usr/share/sounds/sf2/FluidR3_GM.sf2
```

In the QjackCtl GUI, click the "Connect" button. In the "Audio" tab, select your
synthesizer from the list on the left (e.g., "fluidsynth") and select "system"
from the list on the right. Then click the "Connect" button at the bottom.

You can now build the demo with:

```bash
$ bazel build //magenta/interfaces/midi:midi
```

Once built, run this command:

```bash
$ bazel-bin/magenta/interfaces/midi/midi --list
```

You should see a list of available input and output ports, including both the
controller (e.g., "vkeybd") and synthesizer (e.g., "fluidsynth").

You can now start the interface with this command:

```bash
$ bazel-bin/magenta/interfaces/midi/midi \
  --input_port=<controller port> \
  --output_port=<synthesizer port> \
  --bpm=90
```

To initialize a capture session, you need to send the appropriate control change
message from the controller. By default, this is done by setting the modulation
wheel to its max value.

If using vkeybd, you can adjust the modulation wheel value by selecting
"Controls" from the "View" menu and using the slider.

You should immediately hear a metronome and the keys will now produce sounds
through your audio output.

When you have played your priming sequence, end the capture session by sending
the appropriate control change message from the controller. By default, this is
done by setting the modulation wheel back to 0.

After a very short delay, you will hear the input sequence followed by the
generated sequence. You can continue to switch between capture and generating
states using the appropriate control (e.g., the modulation wheel).

## Changing the Capture/Generate Toggle

You can remap the control signals to use something other than the modulation
wheel (e.g., physical pads on your controller). This is done by setting the
`--start_capture_control_*` and `--stop_capture_control_*` flags appropriately.