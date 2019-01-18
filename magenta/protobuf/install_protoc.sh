#!/bin/bash
# Install the .protoc compiler
curl -o /tmp/protoc3.zip -L https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-${1}-x86_64.zip

# Unzip
unzip -d /tmp/protoc3 /tmp/protoc3.zip

# Move protoc to /usr/local/bin/
sudo mv /tmp/protoc3/bin/* /usr/local/bin/

# Move protoc3/include to /usr/local/include/
sudo mv /tmp/protoc3/include/* /usr/local/include/
