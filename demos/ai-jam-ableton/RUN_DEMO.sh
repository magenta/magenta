# Magenta NIPS Demo 2016
# Requirements:
# - MaxMSP 7 (MIDI Control)
# - Mira iPad App (iPad UI)
# - Magenta (MIDI Generation)
# - Ableton Live 9 Suite (Sound Generation)

open NIPS_2016_Demo.als
open magenta_mira.maxpat

magenta_midi \
    --input_port="IAC Driver IAC Bus 1" \
    --output_port="IAC Driver IAC Bus 2" \
    --passthrough=false \
    --qpm=120 \
    --allow_overlap=true \
    --enable_metronome=false \
    --log=DEBUG \
    --clock_control_number=1 \
    --end_call_control_number=2 \
    --min_listen_ticks_control_number=3 \
    --max_listen_ticks_control_number=4 \
    --response_ticks_control_number=5 \
    --temperature_control_number=6 \
    --tempo_control_number=7 \
    --generator_select_control_number=8 \
    --state_control_number=9 \
    --loop_control_number=10 \
    --panic_control_number=11 \
    --mutate_control_number=12 \
    --bundle_files=./basic_rnn.mag,./lookback_rnn.mag,./attention_rnn.mag,./rl_rnn.mag,./polyphony_rnn.mag,./pianoroll_rnn_nade.mag \
    --playback_offset=-0.035 \
    --playback_channel=1&
MAGENTA_PIANO=$!

magenta_midi \
    --input_port="IAC Driver IAC Bus 3" \
    --output_port="IAC Driver IAC Bus 4" \
    --passthrough=false \
    --qpm=120 \
    --allow_overlap=true \
    --enable_metronome=false \
    --clock_control_number=1 \
    --end_call_control_number=2 \
    --min_listen_ticks_control_number=3 \
    --max_listen_ticks_control_number=4 \
    --response_ticks_control_number=5 \
    --temperature_control_number=6 \
    --tempo_control_number=7 \
    --generator_select_control_number=8 \
    --state_control_number=9 \
    --loop_control_number=10 \
    --panic_control_number=11 \
    --mutate_control_number=12 \
    --bundle_files=./drum_kit_rnn.mag \
    --playback_offset=-0.035 \
    --playback_channel=2 \
    --log=INFO&
MAGENTA_DRUMS=$!

trap "kill ${MAGENTA_PIANO} ${MAGENTA_DRUMS}; exit 1" INT
wait

