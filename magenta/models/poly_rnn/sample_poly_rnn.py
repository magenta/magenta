from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from tfkdllib import numpy_softmax, numpy_sample_softmax
from tfkdllib import duration_and_pitch_to_midi
from tfkdllib import tfrecord_duration_and_pitch_iterator
import graph

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None, 'Polyphonic tfrecord file')

def validate_sample_args(model_ckpt,
                         runtime,
                         sample_path,
                         prime,
                         sample,
                         sample_len,
                         temperature,
                         **kwargs):
    return (model_ckpt, runtime, sample_path, prime, sample, sample_len, temperature)

def sample(kwargs):
    (model_ckpt,
     runtime,
     sample_path,
     prime,
     sample,
     sample_len,
     temperature) = validate_sample_args(**kwargs)
    g = graph.Graph(FLAGS.input)
    g.valid_itr.reset()
    duration_mb, note_mb = g.valid_itr.next()
    duration_and_pitch_to_midi(sample_path + "/gt_%i.mid" % runtime, duration_mb[:, 0], note_mb[:, 0])

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        model_dir = str(os.sep).join(model_ckpt.split(os.sep)[:-1])
        model_name = model_ckpt.split(os.sep)[-1]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("Unable to restore from checkpoint")
        i_h1 = np.zeros((g.batch_size, g.rnn_dim)).astype("float32")

        prime = 8
        note_mb = note_mb[:prime]
        duration_mb = duration_mb[:prime]
        for n in range(duration_mb.shape[1]):
            duration_and_pitch_to_midi(sample_path + "/pre%i_%i.mid" % (n, runtime),
                                       duration_mb[:, n], note_mb[:, n], prime)

        note_inputs = note_mb
        duration_inputs = duration_mb

        """
        note_inputs = np.zeros((1, batch_size, train_itr.simultaneous_notes))
        duration_inputs = np.zeros((1, batch_size, train_itr.simultaneous_notes))
        """

        shp = note_inputs.shape
        full_notes = np.zeros((sample_len, shp[1], shp[2]), dtype="float32")
        full_notes[:len(note_inputs)] = note_inputs[:]
        shp = duration_inputs.shape
        full_durations = np.zeros((sample_len, shp[1], shp[2]), dtype="float32")
        full_durations[:len(duration_inputs)] = duration_inputs[:]

        random_state = np.random.RandomState(1999)
        for j in range(sample_len - 1):
            # even predictions are note, odd are duration
            for ni in range(2 * g.n_notes):
                feed = {g.note_inpt: full_notes[j][None, :, :],
                        g.note_target: full_notes[j + 1][None, :, :],
                        g.duration_inpt: full_durations[j][None, :, :],
                        g.duration_target: full_durations[j + 1][None, :, :],
                        g.init_h1: i_h1}
                outs = []
                outs += g.note_preds
                outs += g.duration_preds
                outs += [g.final_h1]
                r = sess.run(outs, feed)
                h_l = r[-1:]
                h1_l = h_l[-1]
                this_preds = r[:-1]
                this_probs = [numpy_softmax(p, temperature=temperature)
                              for p in this_preds]
                this_samples = [numpy_sample_softmax(p, random_state)
                                for p in this_probs]
                if j < (len(note_inputs) - 1):
                    # bypass sampling for now - still in prime seq
                    continue
                note_probs = this_probs[:g.n_notes]
                duration_probs = this_probs[g.n_notes:]
                si = ni // 2
                if (ni % 2) == 0:
                    # only put the single note in...
                    full_notes[j + 1, :, si] = this_samples[si].ravel()
                if (ni % 2) == 1:
                    full_durations[j + 1, :, si] = this_samples[si + g.n_notes].ravel()
            i_h1 = h1_l

        for n in range(full_durations.shape[1]):
            duration_and_pitch_to_midi(sample_path + "/sampled%i_%i.mid" % (n, runtime),
                                       full_durations[:, n], full_notes[:, n],
                                       prime)


def main(argv):
    # prime is the text to prime with
    # sample is 0 for argmax, 1 for sample per character, 2 to sample per space
    import sys
    if len(argv) < 3:
        import time
        runtime = int(time.time())
    else:
        runtime = int(argv[2])
    if len(argv) < 4:
        sample_path = "samples"
    else:
        sample_path = str(argv[3])
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    kwargs = {"model_ckpt": argv[1],
              "runtime": runtime,
              "sample_path": sample_path,
              "prime": " ",
              "sample": 1,
              "sample_len": 50,
              "temperature": .35}
    sample(kwargs)

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
