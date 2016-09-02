new_http_archive(
  name = "pretty_midi",
  build_file = "pretty_midi.BUILD",
  url = "https://github.com/craffel/pretty-midi/archive/c9700f2f72e1873ff64a4911f46ec04b58dfb2de.tar.gz",
  sha256 = "2a17878b5e07fc2982969cf8a41a16d8bb21c86ecbca5f399d3f2553212c4fff",
  strip_prefix = "pretty-midi-c9700f2f72e1873ff64a4911f46ec04b58dfb2de/pretty_midi",
)

new_http_archive(
  name = "midi",
  build_file = "python_midi.BUILD",
  url = "https://github.com/vishnubob/python-midi/archive/4b7a229f6b340e7424c1fccafa9ac543b9b3d605.tar.gz",
  sha256 = "27dcc9e67db0f3fd56420f5f21c7b70f949716a1cfee4e041cd0b1155cef7c4e",
  strip_prefix = "python-midi-4b7a229f6b340e7424c1fccafa9ac543b9b3d605/src",
)

git_repository(
  name = "protobuf",
  remote = "https://github.com/google/protobuf",
  commit = "18a9140f3308272313a9642af58ab0051ac09fd2",
)

new_http_archive(
  name = "six_archive",
  build_file = "six.BUILD",
  url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
  sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
  strip_prefix = "six-1.10.0"
)

bind(
  name = "six",
  actual = "@six_archive//:six",
)

bind(
    name = "python_headers",
    actual = "//util/python:python_headers",
)

new_http_archive(
    name = "mido",
    build_file = "mido.BUILD",
    url = "https://github.com/olemb/mido/archive/1.1.14.tar.gz",
    sha256 = "eacfae91d2efd59ada6bf27f4548bfc2f2b0e8623c80bf242477314ae8df31b5",
    strip_prefix = "mido-1.1.14/mido",
)

new_http_archive(
    name = "music21",
    build_file = "music21.BUILD",
    url = "https://github.com/cuthbertLab/music21/releases/download/v3.0.3-alpha/music21-3.0.3.tar.gz",
    sha256 = "4c0cc1e1fa3638c53ecf45ec13301174114dcf59f93faffc6586c5b94ae065e3",
    strip_prefix = "music21-3.0.3/music21",
)