new_http_archive(
  name = "pretty_midi",
  build_file = "pretty_midi.BUILD",
  url = "https://github.com/craffel/pretty-midi/archive/0.2.6.tar.gz",
  sha256 = "8326c9c87d5efc91670a5881581eb192b095a1c93afd5fddc91b2232af8e9b9b",
  strip_prefix = "pretty-midi-0.2.6/pretty_midi",
)

new_http_archive(
  name = "music21",
  build_file = "music21.BUILD",
  url = "https://github.com/cuthbertLab/music21/releases/download/v3.0.3-alpha/music21-3.0.3.tar.gz",
  sha256 = "4c0cc1e1fa3638c53ecf45ec13301174114dcf59f93faffc6586c5b94ae065e3",
  strip_prefix = "music21-3.0.3/music21",
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
    url = "https://github.com/olemb/mido/archive/1.1.17.tar.gz",
    sha256 = "7844ff77ab12469504c46e9aa035722a2829e7c72b8b6241c78d356895e88114",
    strip_prefix = "mido-1.1.17/mido",
)
