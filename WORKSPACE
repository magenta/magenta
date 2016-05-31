local_repository(
  name = "tf",
  path = __workspace_dir__ + "/tensorflow",
)

load("//tensorflow/tensorflow:workspace.bzl", "tf_workspace")
tf_workspace("tensorflow/", "@tf")

# Specify the minimum required Bazel version.
load("@tf//tensorflow:tensorflow.bzl", "check_version")
check_version("0.2.0")

new_git_repository(
  name = "pretty_midi",
  build_file = "pretty_midi.BUILD",
  remote = "https://github.com/craffel/pretty-midi.git",
  commit = "bd53b637fe6896aa56ec9174337bbbc56f72f67e",
)
