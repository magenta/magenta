FROM tensorflow/tensorflow:0.10.0rc0-devel

MAINTAINER Curtis Hawthorne <fjord@google.com>

# Download and build Magenta.

ENV MAGENTA_MASTER_REF a6757dc00742b57edc807ca195f75f3d7124caac

WORKDIR /
RUN git clone https://github.com/tensorflow/magenta.git && \
    cd magenta && \
    git reset --hard $MAGENTA_MASTER_REF
WORKDIR /magenta
RUN bazel build //magenta/... && bazel test //magenta/...

# Start an interactive shell

CMD ["bash"]
