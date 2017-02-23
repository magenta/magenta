from magenta.pipelines import pipeline
from magenta.protobuf import music_pb2
import tensorflow as tf

"""
Pipeline that filters out all NoteSequences that don't have a given time signature
"""
class TimeSignatureFilter(pipeline.Pipeline):

	def __init__(self, sig_numerator, sig_denominator, name=None):
			super(TimeSignatureFilter, self).__init__(
					input_type=music_pb2.NoteSequence,
					output_type=music_pb2.NoteSequence,
					name=name)
			self.sig_numerator = sig_numerator
			self.sig_denominator = sig_denominator

	def transform(self, note_sequence):
		sigs = list(note_sequence.time_signatures)
		for sig in sigs:
			if sig.numerator != self.sig_numerator:
				return []
			if sig.denominator != self.sig_denominator:
				return []
		return [note_sequence]

"""
Pipeline that wraps a sequence encoder
"""
class EncoderPipeline(pipeline.Pipeline):

	def __init__(self, input_type, encoder_decoder, name=None):
		super(EncoderPipeline, self).__init__(
				input_type=input_type,
				output_type=tf.train.SequenceExample,
				name=name)
		self._encoder_decoder = encoder_decoder

	# TODO: Eventually, make this work such that 'seq' can contain both timeslices
	#    and condition dicts.
	def transform(self, seq):
		encoded = self._encoder_decoder.encode(seq)
		return [encoded]