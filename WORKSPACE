local_repository(
  name = "tf",
  path = __workspace_dir__ + "/tensorflow",
)

load('//tensorflow/tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace("tensorflow/", "@tf")
