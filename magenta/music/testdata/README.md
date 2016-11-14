This readme describes how the files in this testdata folder was created.

* bach-one_phrase-4_voices.xml: Consists of the first phrase of Aus meines
    Herzens Grunde manually encoded in MuseScore and then exported as MusicXML.

* bach-one_phrase-note_sequence.tfrecord: This TFRecord was created by first
    parsing the MusicXML file bach-one_phrase-4_voices.xml using music21, and
    then using magenta.music.pretty_music21 to convert the music21 score object
    into a NoteSequence proto, and then it was written to disk as a TFRecord.
