new_http_archive(
    name = "pretty_midi",
    build_file = "pretty_midi.BUILD",
    sha256 = "8326c9c87d5efc91670a5881581eb192b095a1c93afd5fddc91b2232af8e9b9b",
    strip_prefix = "pretty-midi-0.2.6/pretty_midi",
    url = "https://github.com/craffel/pretty-midi/archive/0.2.6.tar.gz",
)

http_archive(
    name = "protobuf",
    sha256 = "2a25c2b71c707c5552ec9afdfb22532a93a339e1ca5d38f163fe4107af08c54c",
    strip_prefix = "protobuf-3.2.0",
    url = "https://github.com/google/protobuf/archive/v3.2.0.tar.gz",
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
    sha256 = "7844ff77ab12469504c46e9aa035722a2829e7c72b8b6241c78d356895e88114",
    strip_prefix = "mido-1.1.17/mido",
    url = "https://github.com/olemb/mido/archive/1.1.17.tar.gz",
)
