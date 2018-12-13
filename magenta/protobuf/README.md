# Protobuf

This page describe how to update the protobuf generated python file. By
default, the protobuf is already compiled into python file so you won't have to
do anything. Those steps are required only if you update the `.proto` file.

Install the proto compiler (version 3.6.1):

```bash
./install_protoc.sh <linux|osx>
```

Re-generate the python file:

```bash
./generate_pb2_py.sh
```
