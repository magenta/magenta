new_http_archive(
    name = "pretty_midi",
    build_file = "pretty_midi.BUILD",
    sha256 = "f359310473b3e1ed070beda08fbbd7564dc7fed26aec00f06bdb5088394ae4d2",
    strip_prefix = "pretty-midi-0.2.8/pretty_midi",
    url = "https://github.com/craffel/pretty-midi/archive/0.2.8.tar.gz",
)

http_archive(
    name = "protobuf",
    sha256 = "3f833c1c367f53803f5f849181af4a4edb20f8dd1fbdced19b5a2d52ee43ed54",
    strip_prefix = "protobuf-3.4.0",
    url = "https://github.com/google/protobuf/releases/download/v3.4.0/protobuf-python-3.4.0.tar.gz",
)

new_http_archive(
    name = "six_archive",
    build_file = "six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
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
    sha256 = "870d2f470ce1123324f9ef9676b6c9f2580293dd2a07fdfe00e20a47740e8b8e",
    strip_prefix = "mido-1.2.6/mido",
    url = "https://github.com/olemb/mido/archive/1.2.6.tar.gz",
)
