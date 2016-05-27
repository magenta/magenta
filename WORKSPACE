local_repository(
  name = "tf",
  path = __workspace_dir__ + "/tensorflow",
)

load('//tensorflow/tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace("tensorflow/", "@tf")

# Specify the minimum required Bazel version.
load("@tf//tensorflow:tensorflow.bzl", "check_version")
check_version("0.2.0")