new_http_archive(
  name = "pretty_midi",
  build_file = "pretty_midi.BUILD",
  url = "https://github.com/craffel/pretty-midi/archive/9e3c353b7b77ce4012c70610db4c0d0961c05916.zip",
  sha256 = "960edecb4b5b2ae25f68cd95ab579027d0537975fe32ef0f82c173266beeec77",
  strip_prefix = "pretty-midi-9e3c353b7b77ce4012c70610db4c0d0961c05916/pretty_midi",
)

new_http_archive(
  name = "midi",
  build_file = "python_midi.BUILD",
  url = "https://github.com/vishnubob/python-midi/archive/4b7a229f6b340e7424c1fccafa9ac543b9b3d605.zip",
  sha256 = "37699e01cebf304a223f91a576f70c654c26545edc932671e05ec186d0970731",
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