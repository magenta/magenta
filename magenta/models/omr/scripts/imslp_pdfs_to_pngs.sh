#!/bin/bash
# Extracts PNGs from the IMSLP backup torrent:
# http://imslp.org/images/imslpbackup.torrent
# The output PNGs are suitable for training and running OMR. More details on the
# torrent here: http://imslp.org/wiki/IMSLP:Backups
# Note: We believe, but are unable to verify, that the contents of the torrent
# are public domain in the United States.

TORRENT_DIR="$1"
export OUTPUT_DIR="$2"

if ! [[ -d "$TORRENT_DIR" ]]; then
  echo "First argument must be a directory" > /dev/stderr
  exit -1
fi

if ! [[ -d "$OUTPUT_DIR" ]]; then
  mkdir -v "$OUTPUT_DIR"
fi

if ! [[ -x "$(which pdfimages)" ]]; then
  echo "pdfimages is required. Please install poppler-utils." > /dev/stderr
  exit -1
fi

if ! [[ -x "$(which parallel)" ]]; then
  echo "GNU parallel is required. Please install parallel." > /dev/stderr
  exit -1
fi

if ! [[ -x "$(which convert)" ]]; then
  echo "'convert' is required. Please install imagemagick." > /dev/stderr
  exit -1
fi

# Convert each IMSLP PDF to "IMSLPnnnnn-nnn.ppm" or ".pgm" images in $OUTPUT_DIR.
perl -w /dev/stdin $(find "$TORRENT_DIR" -name "IMSLP*.pdf") <<'END' | parallel -v
for (@ARGV) {
  chomp;
  /(IMSLP([0-9]+))/;
  # Images are sharded--IMSLP IDs are split into buckets of 1000.
  my $dir = $ENV{"OUTPUT_DIR"} . "/" . int($2 / 1000);
  print qq(mkdir -p $dir\npdfimages "$_" $dir/$1\n);
}
END

# Convert extracted "ppm" and "pgm" images to PNG.
(for file in "$OUTPUT_DIR"/*/*.p[pg]m; do
  echo "convert '$file' '${file%.*}.png' && rm -v '$file'"
done) | parallel -v
