FROM tensorflow/tensorflow:0.10.0-devel

MAINTAINER Curtis Hawthorne <fjord@google.com>

# Download and build Magenta.

ENV MAGENTA_MASTER_REF v0.1.3

WORKDIR /
RUN git clone https://github.com/tensorflow/magenta.git && \
    cd magenta && \
    git reset --hard $MAGENTA_MASTER_REF
WORKDIR /magenta
RUN bazel build //magenta/... && bazel test //magenta/...

# Required to build the pip package
RUN pip install pretty_midi

RUN bazel build magenta/tools/pip:build_pip_package && \
    bazel-bin/magenta/tools/pip/build_pip_package /tmp/magenta_pkg && \
    pip install --upgrade /tmp/magenta_pkg/magenta-*.whl

# Add pre-trained models
ADD http://download.magenta.tensorflow.org/models/lookback_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/attention_rnn.mag /magenta-models/
ADD http://download.magenta.tensorflow.org/models/basic_rnn.mag /magenta-models/

# /magenta-data should be mapped to the host on startup.
RUN mkdir /magenta-data
WORKDIR /magenta-data

# Start an interactive shell

CMD ["bash"]
