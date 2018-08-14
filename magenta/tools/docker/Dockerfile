FROM tensorflow/tensorflow:latest-devel

MAINTAINER Curtis Hawthorne <fjord@google.com>

# Download and build Magenta.

ENV MAGENTA_MASTER_REF v0.3.11

# Required for local development
RUN pip install scipy matplotlib intervaltree bokeh IPython librosa mir_eval
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential libasound2-dev libjack-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN git clone https://github.com/tensorflow/magenta.git && \
    cd magenta && \
    git reset --hard $MAGENTA_MASTER_REF
WORKDIR /magenta
RUN bazel build //magenta/... && bazel test --test_output=errors //magenta/...

RUN bazel build magenta/tools/pip:build_pip_package && \
    bazel-bin/magenta/tools/pip/build_pip_package /tmp/magenta_pkg && \
    pip install --upgrade /tmp/magenta_pkg/magenta-*.whl

# Add pre-trained models
ADD http://download.magenta.tensorflow.org/models/attention_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/basic_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/chord_pitches_improv.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/drum_kit_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/lookback_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/polyphony_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/rl_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/rl_tuner_note_rnn.ckpt /magenta-models/

ADD http://download.magenta.tensorflow.org/models/multistyle-pastiche-generator-monet.ckpt /magenta-models/
ADD http://download.magenta.tensorflow.org/models/multistyle-pastiche-generator-varied.ckpt /magenta-models/

# /magenta-data should be mapped to the host on startup.
RUN mkdir /magenta-data
WORKDIR /magenta-data

# Start an interactive shell

CMD ["bash"]
