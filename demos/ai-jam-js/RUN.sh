
#!/bin/bash

python server/server.py &
WEB_SERVER=$!

magenta_clock\
  --output_port="magenta_clock" \
  --output_channels=0 \
  --clock_control_number=42 \
  --log=INFO &
MAGENTA_CLOCK=$!

magenta_midi \
  --input_ports="magenta_drums_in,magenta_clock" \
  --output_port="magenta_out" \
  --bundle_files=./drum_kit_rnn.mag\
  --playback_channel=9 \
  --passthrough=False \
  --loop_control_number=1 \
  --mutate_control_number=2 \
  --temperature_control_number=3 \
  --response_ticks_control_number=4 \
  --max_listen_ticks_control_number=5 \
  --panic_control_number=6 \
  --clock_control_number=42 \
  --log=INFO &
MAGENTA_DRUMS=$!

magenta_midi \
  --input_port="magenta_piano_in,magenta_clock" \
  --output_port="magenta_out" \
  --bundle_files=./pianoroll_rnn_nade.mag,./lookback_rnn.mag,./attention_rnn.mag\
  --capture_channel=0 \
  --playback_channel=0 \
  --passthrough=False \
  --generator_select_control_number=0 \
  --loop_control_number=1 \
  --mutate_control_number=2 \
  --temperature_control_number=3 \
  --response_ticks_control_number=4 \
  --max_listen_ticks_control_number=5 \
  --panic_control_number=6 \
  --clock_control_number=42 \
  --log=INFO &
MAGENTA_PIANO=$!

sleep 20

open -a "Google Chrome" "http://localhost:8080"

trap "kill ${WEB_SERVER} ${MAGENTA_CLOCK} ${MAGENTA_PIANO} ${MAGENTA_DRUMS}; exit 1" INT
wait
