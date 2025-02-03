# Open source code for ReaLchords and ReaLJam

## Load and use the model

```python
import magenta

import gin
import jax.numpy as jnp
import note_seq
import numpy as np
from colabtools import sound
import note_seq

import json

from magenta.models.realchords import frame_codec
from magenta.models.realchords import vocab
from magenta.models.realchords import inference_utils

base_model_gin_file_path = 'realchords_opensource_gin/small_model.gin'
inference_with_data_model_file_path = 'realchords_opensource_gin/inference_with_data_small_model.gin'
gin.add_config_file_search_path('')
batch_size = 128
model_part = 'chord'
order = 'causal'
seed = 1234
MODEL_STEP=20000

def generate_with_chord_causal_model(with_data_model, tokens_gt_all):
  tokens_causal_all = []

  for tokens_gt in tqdm(tokens_gt_all, total=len(tokens_gt_all)):

    # We pad the data length to 512 and truncate back to original shape because the model is compiled to see the fixed data shape.
    current_batch_size, data_length = jnp.array(tokens_gt).shape
    midi_tokens = np.pad(
            jnp.array(tokens_gt),
            [[0, 0], [0, 512-data_length]],
            constant_values=0,
        )
    if current_batch_size < batch_size:
      residual_batch_size = batch_size-current_batch_size
      midi_tokens = np.pad(midi_tokens,[[0, residual_batch_size], [0, 0]], constant_values=0)
    else:
      residual_batch_size = 0
    tokens_causal_model = with_data_model(decoder_prompt=jnp.zeros([batch_size, 0], dtype=np.int32),decode_length=512, midi_tokens = midi_tokens)
    if residual_batch_size > 0:
      tokens_causal_all.append(np.array(tokens_causal_model[:-residual_batch_size]))
    else:
      tokens_causal_all.append(np.array(tokens_causal_model))
  tokens_causal_all = jnp.concatenate(tokens_causal_all, axis=0)
  return tokens_causal_all


with_data_chord_model = GenWithDataInferenceModel(
    model_paths['student'],
    'realchords_opensource_gin/base_model.gin',
    MODEL_STEP,
    batch_size=batch_size,
    model_parallel_submesh=(1, 1, 1, 1),
    load_gin_from_model_dir = False,
    gin_overrides = "TASK_FEATURE_LENGTHS={'inputs': 0, 'targets': 512}",
    additional_gin_path = ['realchords_opensource_gin/inference_with_data_model.gin','realchords_opensource_gin/inference_with_data_small_model.gin'],
    )
tokens_causal_all = generate_with_chord_causal_model(with_data_chord_model, tokens_gt_all_batched)

def generate_with_chord_non_causal_model(non_causal_model, eval_ds_causal):
  tokens_non_causal_input_all = []
  tokens_non_causal_output_gt_all = []
  tokens_non_causal_output_model_all = []

  for batch in tqdm(eval_ds_causal):
    causal_tokens = batch['targets']
    inputs, targets = sequence_utils.causal_to_enc_dec_np(
        causal_tokens
    )
    inputs = sequence_utils.add_eos(inputs)
    targets = sequence_utils.add_eos(targets)
    # Although here we provided ground-truth decoder_target_tokens,
    # in predict_batch_with_aux, by default it will inference from scratch.
    batch = {
        'encoder_input_tokens': inputs,
        'decoder_input_tokens': targets,
    }
    tokens_non_causal_input = batch['encoder_input_tokens']
    tokens_non_causal_input_all.append(jnp.array(tokens_non_causal_input))
    tokens_non_causal_output_gt_all.append(jnp.array(batch['decoder_input_tokens']))

    # We pad the data length to 256 and truncate back to original shape because the model is compiled to see the fixed data shape.
    current_batch_size, data_length = jnp.array(tokens_non_causal_input).shape

    input_tokens = np.pad(
            jnp.array(tokens_non_causal_input),
            [[0, 0], [0, 256-data_length]],
            constant_values=0,
        )
    target_tokens = np.pad(
            jnp.array(targets),
            [[0, 0], [0, 256-data_length]],
            constant_values=0,
        )

    if current_batch_size < batch_size:
      residual_batch_size = batch_size-current_batch_size
      input_tokens = np.pad(input_tokens,[[0, residual_batch_size], [0, 0]], constant_values=0)
      target_tokens = np.pad(target_tokens,[[0, residual_batch_size], [0, 0]], constant_values=0)
    else:
      residual_batch_size = 0

    tokens_non_causal_model = non_causal_model(
        decoder_prompt=jnp.zeros([batch_size, 0], dtype=np.int32),
        decode_length=256,
        encoder_input_tokens = input_tokens,
        seed=1234)

    tokens_non_causal_model *= (target_tokens > 1).astype(jnp.int32)
    causal_tokens = jnp.zeros(
        [tokens_non_causal_model.shape[0], tokens_non_causal_model.shape[1] * 2], dtype=jnp.int32
    )
    causal_tokens = causal_tokens.at[:, ::2].set(tokens_non_causal_model)
    causal_tokens = causal_tokens.at[:, 1::2].set(input_tokens * (input_tokens > 1).astype(jnp.int32))
    causal_tokens *= (causal_tokens > 1).astype(jnp.int32) # no eos
    if residual_batch_size > 0:
      tokens_non_causal_output_model_all.append(np.array(causal_tokens)[:-residual_batch_size])
    else:
      tokens_non_causal_output_model_all.append(np.array(causal_tokens))

  tokens_non_causal_input_all = jnp.concatenate(tokens_non_causal_input_all, axis=0)
  tokens_non_causal_output_gt_all = jnp.concatenate(tokens_non_causal_output_gt_all, axis=0)
  tokens_non_causal_output_model_all = jnp.concatenate(tokens_non_causal_output_model_all, axis=0)
  return tokens_non_causal_input_all, tokens_non_causal_output_gt_all, tokens_non_causal_output_model_all

chord_non_causal_model = NonCausalMusicTransformerInferenceModel(
    non_causal_chord_model_path,
    'realchords_opensource_gin/base_model_non_causal.gin',
    pretrain_model_step,
    batch_size=batch_size,
    model_parallel_submesh=(1, 1, 1, 1),
    load_gin_from_model_dir = False,
    additional_gin_path = ['realchords_opensource_gin/base_model_non_causal.gin'],
    gin_overrides = "TASK_FEATURE_LENGTHS={'inputs': 0, 'targets': 512} \n TRAIN_TASK_FEATURE_LENGTHS={'inputs': 0, 'targets': 512} \n",
    )

def show_and_play_song(song,chord_mapping):

  note_seq_midi = summaries.song_to_note_sequence(song, chord_mapping)
  note_seq.plot_sequence(note_seq_midi)
  audio = summaries.synthesize_note_sequences([note_seq_midi], sample_rate=16000, num_seconds=30)[0]
  sound.Play(audio, 16000)

def postprocess_causal(tokens_causal_all):
  codec = frame_codec.get_frame_codec()
  model_part = 'chord'
  order = 'causal'

  num_data = tokens_gt_all.shape[0]

  songs_causal_all = []
  for i in tqdm(range(num_data)):
    tokens_causal = tokens_causal_all[i]
    pad_mask = tokens_causal>1
    songs_causal_all.append(
        frame_codec.postprocess_frame_example(
            codec=codec,
            tokens=tokens_causal[pad_mask],
            example=None,
            model_part=model_part,
            order=order)['song']
        )
  return songs_causal_all
```

## Start the server
TODO