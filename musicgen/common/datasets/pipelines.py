from magenta.pipelines import pipeline
from magenta.pipelines import statistics
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
			if (sig.numerator != self.sig_numerator) or (sig.denominator != self.sig_denominator):
				tf.logging.warning('Filtering out note sequence with time signature = %d/%d',
					sig.numerator, sig.denominator)
				self._set_stats([statistics.Counter(
					'sequences_discarded_because_wrong_time_signature', 1)])
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


"""
Catchall pipeline that allows arbitrary behavior given by the input 'transformFn' function
"""
class CustomFunctionPipeline(pipeline.Pipeline):

	def __init__(self, input_type, output_type, transformFn, name=None):
		super(CustomFunctionPipeline, self).__init__(
				input_type=input_type,
				output_type=output_type,
				name=name)
		self.transformFn = transformFn

	def transform(self, seq):
		return self.transformFn(seq)

