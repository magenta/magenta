FROM tensorflow/tensorflow:0.10.0-devel

MAINTAINER Curtis Hawthorne <fjord@google.com>

# Download and build Magenta.

ENV MAGENTA_MASTER_REF ec86d0a57d274ac3b20d3d467098e91651107ac3

WORKDIR /
RUN git clone https://github.com/tensorflow/magenta.git && \
    cd magenta && \
    git reset --hard $MAGENTA_MASTER_REF
WORKDIR /magenta
RUN bazel build //magenta/... && bazel test //magenta/...

# Add pre-trained models
ADD http://download.magenta.tensorflow.org/models/lookback_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/attention_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/basic_rnn.mag /magenta-models/

# Start an interactive shell

CMD ["bash"]
