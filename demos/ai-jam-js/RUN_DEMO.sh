#!/bin/bash

python maybe_download_mags.py

cd server
python server.py &
WEB_SERVER=$!
cd ..

midi_clock\
  --output_port="magenta_clock" \
  --output_channels=0 \
  --clock_control_number=1 \
  --log=INFO &
MIDI_CLOCK=$!

magenta_midi \
  --input_ports="magenta_drums_in,magenta_clock" \
  --output_port="magenta_out" \
  --bundle_files=./drum_kit_rnn.mag\
  --qpm=120 \
  --allow_overlap=true \
  --playback_channel=9 \
  --enable_metronome=false \
  --passthrough=false \
  --clock_control_number=1 \
  --min_listen_ticks_control_number=3 \
  --max_listen_ticks_control_number=4 \
  --response_ticks_control_number=5 \
  --temperature_control_number=6 \
  --generator_select_control_number=8 \
  --loop_control_number=10 \
  --panic_control_number=11 \
  --mutate_control_number=12 \
  --log=INFO &
MAGENTA_DRUMS=$!

magenta_midi \
  --input_port="magenta_piano_in,magenta_clock" \
  --output_port="magenta_out" \
  --bundle_files=./attention_rnn.mag,./pianoroll_rnn_nade.mag,./performance.mag \
  --qpm=120 \
  --allow_overlap=true \
  --playback_channel=0 \
  --enable_metronome=false \
  --passthrough=false \
  --generator_select_control_number=0 \
  --clock_control_number=1 \
  --min_listen_ticks_control_number=3 \
  --max_listen_ticks_control_number=4 \
  --response_ticks_control_number=5 \
  --temperature_control_number=6 \
  --generator_select_control_number=8 \
  --loop_control_number=10 \
  --panic_control_number=11 \
  --mutate_control_number=12 \
  --log=INFO &
MAGENTA_PIANO=$!

trap "kill ${WEB_SERVER} ${MIDI_CLOCK} ${MAGENTA_PIANO} ${MAGENTA_DRUMS}; exit 1" INT

sleep 20

open -a "Google Chrome" "http://localhost:8080"

wait
