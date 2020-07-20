# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Defines the Material Design Icons Problem."""
import io
import numpy as np

from PIL import Image
from six.moves import zip_longest
from skimage import draw

import tensorflow.compat.v1 as tf


SVG_PREFIX_BIG = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="'
                  'http://www.w3.org/1999/xlink" width="256px" height="256px"'
                  ' style="-ms-transform: rotate(360deg); -webkit-transform:'
                  ' rotate(360deg); transform: rotate(360deg);" '
                  'preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24">')
PATH_PREFIX_1 = '<path d="'
PATH_POSFIX_1 = '" fill="currentColor"/>'
SVG_POSFIX = '</svg>'

NUM_ARGS = {'v': 1, 'V': 1, 'h': 1, 'H': 1, 'a': 7, 'A': 7, 'l': 2, 'L': 2,
            't': 2, 'T': 2, 'c': 6, 'C': 6, 'm': 2, 'M': 2, 's': 4, 'S': 4,
            'q': 4, 'Q': 4, 'z': 0}
# in order of arg complexity, with absolutes clustered
# recall we don't handle all commands (see docstring)
CMDS_LIST = 'zhvmltsqcaHVMLTSQCA'
CMD_MAPPING = {cmd: i for i, cmd in enumerate(CMDS_LIST)}


############################### GENERAL UTILS #################################
def grouper(iterable, batch_size, fill_value=None):
  """Helper method for returning batches of size batch_size of a dataset."""
  # grouper('ABCDEF', 3) -> 'ABC', 'DEF'
  args = [iter(iterable)] * batch_size
  return zip_longest(*args, fillvalue=fill_value)


def map_uni_to_alphanum(uni):
  """Maps [0-9 A-Z a-z] to numbers 0-62."""
  if 48 <= uni <= 57:
    return uni - 48
  elif 65 <= uni <= 90:
    return uni - 65 + 10
  return uni - 97 + 36


############# UTILS FOR CONVERTING SFD/SPLINESETS TO SVG PATHS ################
def _get_spline(sfd):
  if 'SplineSet' not in sfd:
    return ''
  pro = sfd[sfd.index('SplineSet')+10:]
  pro = pro[:pro.index('EndSplineSet')]
  return pro


def _spline_to_path_list(spline, height, replace_with_prev=False):
  """Converts SplineSet to a list of tokenized commands in svg path."""
  path = []
  prev_xy = []
  for line in spline.splitlines():
    if not line:
      continue
    tokens = line.split(' ')
    cmd = tokens[-2]
    if cmd not in 'cml':
      # COMMAND NOT RECOGNIZED.
      return []
      # assert cmd in 'cml', 'Command not recognized: {}'.format(cmd)
    args = tokens[:-2]
    args = [float(x) for x in args if x]

    if replace_with_prev and cmd in 'c':
      args[:2] = prev_xy
    prev_xy = args[-2:]

    new_y_args = []
    for i, a in enumerate(args):
      if i % 2 == 1:
        new_y_args.append((height - a))
      else:
        new_y_args.append((a))

    path.append([cmd.upper()] + new_y_args)
  return path


def sfd_to_path_list(single, replace_with_prev=False):
  """Converts the given SFD glyph into a path."""
  return _spline_to_path_list(_get_spline(single['sfd']),
                              single['vwidth'], replace_with_prev)


#################### UTILS FOR PROCESSING TOKENIZED PATHS #####################
def add_missing_cmds(path, remove_zs=False):
  """Adds missing cmd tags to the commands in the svg."""
  # For instance, the command 'a' takes 7 arguments, but some SVGs declare:
  #   a 1 2 3 4 5 6 7 8 9 10 11 12 13 14
  # Which is 14 arguments. This function converts the above to the equivalent:
  #   a 1 2 3 4 5 6 7  a 8 9 10 11 12 13 14
  #
  # Note: if remove_zs is True, this also removes any occurences of z commands.
  new_path = []
  for cmd in path:
    if not remove_zs or cmd[0] not in 'Zz':
      for new_cmd in add_missing_cmd(cmd):
        new_path.append(new_cmd)
  return new_path


def add_missing_cmd(command_list):
  """Adds missing cmd tags to the given command list."""
  # E.g.: given:
  #   ['a', '0', '0', '0', '0', '0', '0', '0',
  #         '0', '0', '0', '0', '0', '0', '0']
  # Converts to:
  #   [['a', '0', '0', '0', '0', '0', '0', '0'],
  #    ['a', '0', '0', '0', '0', '0', '0', '0']]
  # And returns a string that joins these elements with spaces.
  cmd_tag = command_list[0]
  args = command_list[1:]

  final_cmds = []
  for arg_batch in grouper(args, NUM_ARGS[cmd_tag]):
    final_cmds.append([cmd_tag] + list(arg_batch))

  if not final_cmds:
    # command has no args (e.g.: 'z')
    final_cmds = [[cmd_tag]]

  return final_cmds


def _normalize_args(arglist, norm, add=None, flip=False):
  """Normalize the given args with the given norm value."""
  new_arglist = []
  for i, arg in enumerate(arglist):
    new_arg = float(arg)

    if add is not None:
      add_to_x, add_to_y = add

      # This argument is an x-coordinate if even, y-coordinate if odd
      # except when flip == True
      if i % 2 == 0:
        new_arg += add_to_y if flip else add_to_x
      else:
        new_arg += add_to_x if flip else add_to_y

    new_arglist.append(str(24 * new_arg / norm))
  return new_arglist


def normalize_based_on_viewbox(path, viewbox):
  """Normalizes all args in a path to a standard 24x24 viewbox."""
  # Each SVG lives in a 2D plane. The viewbox determines the region of that
  # plane that gets rendered. For instance, some designers may work with a
  # viewbox that's 24x24, others with one that's 100x100, etc.

  # Suppose I design the the letter "h" in the Arial style using a 100x100
  # viewbox (let's call it icon A). Let's suppose the icon has height 75. Then,
  # I design the same character using a 20x20 viewbox (call this icon B), with
  # height 15 (=75% of 20). This means that, when rendered, both icons with look
  # exactly the same, but the scale of the commands each icon is using is
  # different. For instance, if icon A has a command like "lineTo 100 100", the
  # equivalent command in icon B will be "lineTo 20 20".

  # In order to avoid this problem and bring all real values to the same scale,
  # I scale all icons' commands to use a 24x24 viewbox. This function does this:
  # it converts a path that exists in the given viewbox into a standard 24x24
  # viewbox.
  viewbox = viewbox.split(' ')
  norm = max(int(viewbox[-1]), int(viewbox[-2]))

  if int(viewbox[-1]) > int(viewbox[-2]):
    add_to_y = 0
    add_to_x = abs(int(viewbox[-1]) - int(viewbox[-2])) / 2
  else:
    add_to_y = abs(int(viewbox[-1]) - int(viewbox[-2])) / 2
    add_to_x = 0

  new_path = []
  for command in path:
    if command[0] == 'a':
      new_path.append([command[0]] +
                      _normalize_args(command[1:3], norm) +
                      command[3:6] +
                      _normalize_args(command[6:], norm))
    elif command[0] == 'A':
      new_path.append(
          [command[0]] +
          _normalize_args(command[1:3], norm) +
          command[3:6] +
          _normalize_args(command[6:], norm, add=(add_to_x, add_to_y)))
    elif command[0] == 'V':
      new_path.append(
          [command[0]] +
          _normalize_args(command[1:], norm, add=(add_to_x, add_to_y),
                          flip=True))
    elif command[0] == command[0].upper():
      new_path.append(
          [command[0]] +
          _normalize_args(command[1:], norm, add=(add_to_x, add_to_y)))
    elif command[0] in 'zZ':
      new_path.append([command[0]])
    else:
      new_path.append([command[0]] + _normalize_args(command[1:], norm))

  return new_path


def _convert_args(args, curr_pos, cmd):
  """Converts given args to relative values."""
  # NOTE: glyphs only use a very small subset of commands (L, C, M, and Z -- I
  # believe). So I'm not handling A and H for now.
  if cmd in 'AH':
    raise NotImplementedError('These commands have >6 args (not supported).')

  new_args = []
  for i, arg in enumerate(args):
    x_or_y = i % 2
    if cmd == 'H':
      x_or_y = (i + 1) % 2
    new_args.append(str(float(arg) - curr_pos[x_or_y]))

  return new_args


def _update_curr_pos(curr_pos, cmd, start_of_path):
  """Calculate the position of the pen after cmd is applied."""
  if cmd[0] in 'ml':
    curr_pos = [curr_pos[0] + float(cmd[1]), curr_pos[1] + float(cmd[2])]
    if cmd[0] == 'm':
      start_of_path = curr_pos
  elif cmd[0] in 'z':
    curr_pos = start_of_path
  elif cmd[0] in 'h':
    curr_pos = [curr_pos[0] + float(cmd[1]), curr_pos[1]]
  elif cmd[0] in 'v':
    curr_pos = [curr_pos[0], curr_pos[1] + float(cmd[1])]
  elif cmd[0] in 'ctsqa':
    curr_pos = [curr_pos[0] + float(cmd[-2]), curr_pos[1] + float(cmd[-1])]

  return curr_pos, start_of_path


def make_relative(cmds):
  """Convert commands in a path to relative positioning."""
  curr_pos = (0.0, 0.0)
  start_of_path = (0.0, 0.0)
  new_cmds = []
  for cmd in cmds:
    if cmd[0].lower() == cmd[0]:
      new_cmd = cmd
    elif cmd[0].lower() == 'z':
      new_cmd = [cmd[0].lower()]
    else:
      new_cmd = [cmd[0].lower()] + _convert_args(cmd[1:], curr_pos, cmd=cmd[0])
    new_cmds.append(new_cmd)
    curr_pos, start_of_path = _update_curr_pos(curr_pos, new_cmd, start_of_path)
  return new_cmds


def _is_to_left_of(pt1, pt2):
  pt1_norm = (pt1[0]**2 + pt1[1]**2)
  pt2_norm = (pt2[0]**2 + pt2[1]**2)
  return pt1[1] < pt2[1] or (pt1_norm == pt2_norm and pt1[0] < pt2[0])


def _get_leftmost_point(path):
  """Returns the leftmost, topmost point of the path."""
  leftmost = (float('inf'), float('inf'))
  idx = -1

  for i, cmd in enumerate(path):
    if len(cmd) > 1:
      endpoint = cmd[-2:]
      if _is_to_left_of(endpoint, leftmost):
        leftmost = endpoint
        idx = i

  return leftmost, idx


def _separate_substructures(path):
  """Returns a list of subpaths, each representing substructures the glyph."""
  substructures = []
  curr = []
  for cmd in path:
    if cmd[0] in 'mM' and curr:
      substructures.append(curr)
      curr = []
    curr.append(cmd)
  if curr:
    substructures.append(curr)
  return substructures


def _is_clockwise(subpath):
  """Returns whether the given subpath is clockwise-oriented."""
  pts = [cmd[-2:] for cmd in subpath]
  det = 0
  for i in range(len(pts)-1):
    det += np.linalg.det(pts[i:i+2])
  return det > 0


def _make_clockwise(subpath):
  """Inverts the cardinality of the given subpath."""
  new_path = [subpath[0]]
  other_cmds = list(reversed(subpath[1:]))
  for i, cmd in enumerate(other_cmds):
    if i+1 == len(other_cmds):
      where_we_were = subpath[0][-2:]
    else:
      where_we_were = other_cmds[i+1][-2:]

    if len(cmd) > 3:
      new_cmd = [cmd[0], cmd[3], cmd[4], cmd[1], cmd[2],
                 where_we_were[0], where_we_were[1]]
    else:
      new_cmd = [cmd[0], where_we_were[0], where_we_were[1]]

    new_path.append(new_cmd)
  return new_path


def canonicalize(path):
  """Makes all paths start at top left, and go clockwise first."""
  # convert args to floats
  path = [[x[0]] + list(map(float, x[1:])) for x in path]

  # canonicalize each subpath separately
  new_substructures = []
  for subpath in _separate_substructures(path):
    leftmost_point, leftmost_idx = _get_leftmost_point(subpath)
    reordered = ([['M', leftmost_point[0], leftmost_point[1]]] +
                 subpath[leftmost_idx+1:] + subpath[1:leftmost_idx+1])
    new_substructures.append((reordered, leftmost_point))

  new_path = []
  first_substructure_done = False
  should_flip_cardinality = False
  for sp, _ in sorted(new_substructures, key=lambda x: (x[1][1], x[1][0])):
    if not first_substructure_done:
      # we're looking at the first substructure now, we can determine whether we
      # will flip the cardniality of the whole icon or not
      should_flip_cardinality = not _is_clockwise(sp)
      first_substructure_done = True

    if should_flip_cardinality:
      sp = _make_clockwise(sp)

    new_path.extend(sp)

  # convert args to strs
  path = [[x[0]] + list(map(str, x[1:])) for x in new_path]
  return path


########## UTILS FOR CONVERTING TOKENIZED PATHS TO VECTORS ###########
def path_to_vector(path, categorical=False):
  """Converts path's commands to a series of vectors."""
  # Notes:
  #   - The SimpleSVG dataset does not have any 't', 'q', 'Z', 'T', or 'Q'.
  #     Thus, we don't handle those here.
  #   - We also removed all 'z's.
  #   - The x-axis-rotation argument to a commands is always 0 in this
  #     dataset, so we ignore it

  # Many commands have args that correspond to args in other commands.
  #   v  __,__ _______________ ______________,_________ __,__ __,__ _,y
  #   h  __,__ _______________ ______________,_________ __,__ __,__ x,_
  #   z  __,__ _______________ ______________,_________ __,__ __,__ _,_
  #   a  rx,ry x-axis-rotation large-arc-flag,sweepflag __,__ __,__ x,y
  #   l  __,__ _______________ ______________,_________ __,__ __,__ x,y
  #   c  __,__ _______________ ______________,_________ x1,y1 x2,y2 x,y
  #   m  __,__ _______________ ______________,_________ __,__ __,__ x,y
  #   s  __,__ _______________ ______________,_________ __,__ x2,y2 x,y

  # So each command will be converted to a vector where the dimension is the
  # minimal number of arguments to all commands:
  #   [rx, ry, large-arc-flag, sweepflag, x1, y1, x2, y2, x, y]
  # If a command does not output a certain arg, it is set to 0.
  #   "l 5,5" becomes [0, 0, 0, 0, 0, 0, 0, 0, 5, 5]

  # Also note, as of now we also output an extra dimension at index 0, which
  # indicates which command is being outputted (integer).
  new_path = []
  for cmd in path:
    new_path.append(cmd_to_vector(cmd, categorical=categorical))
  return new_path


def cmd_to_vector(cmd_list, categorical=False):
  """Converts the given command (given as a list) into a vector."""
  # For description of how this conversion happens, see
  # path_to_vector docstring.
  cmd = cmd_list[0]
  args = cmd_list[1:]

  if not categorical:
    # integer, for MSE
    command = [float(CMD_MAPPING[cmd])]
  else:
    # one hot + 1 dim for EOS.
    command = [0.0] * (len(CMDS_LIST) + 1)
    command[CMD_MAPPING[cmd] + 1] = 1.0

  arguments = [0.0] * 10
  if cmd in 'hH':
    arguments[8] = float(args[0])  # x
  elif cmd in 'vV':
    arguments[9] = float(args[0])  # y
  elif cmd in 'mMlLtT':
    arguments[8] = float(args[0])  # x
    arguments[9] = float(args[1])  # y
  elif cmd in 'sSqQ':
    arguments[6] = float(args[0])  # x2
    arguments[7] = float(args[1])  # y2
    arguments[8] = float(args[2])  # x
    arguments[9] = float(args[3])  # y
  elif cmd in 'cC':
    arguments[4] = float(args[0])  # x1
    arguments[5] = float(args[1])  # y1
    arguments[6] = float(args[2])  # x2
    arguments[7] = float(args[3])  # y2
    arguments[8] = float(args[4])  # x
    arguments[9] = float(args[5])  # y
  elif cmd in 'aA':
    arguments[0] = float(args[0])  # rx
    arguments[1] = float(args[1])  # ry
    # we skip x-axis-rotation
    arguments[2] = float(args[3])  # large-arc-flag
    arguments[3] = float(args[4])  # sweep-flag
    # a does not have x1, y1, x2, y2 args
    arguments[8] = float(args[5])  # x
    arguments[9] = float(args[6])  # y

  return command + arguments


################# UTILS FOR RENDERING PATH INTO IMAGE #################
def _cubicbezier(x0, y0, x1, y1, x2, y2, x3, y3, n=40):
  """Return n points along cubiz bezier with given control points."""
  # from http://rosettacode.org/wiki/Bitmap/B%C3%A9zier_curves/Cubic
  pts = []
  for i in range(n+1):
    t = float(i) / float(n)
    a = (1. - t)**3
    b = 3. * t * (1. - t)**2
    c = 3.0 * t**2 * (1.0 - t)
    d = t**3

    x = float(a * x0 + b * x1 + c * x2 + d * x3)
    y = float(a * y0 + b * y1 + c * y2 + d * y3)
    pts.append((x, y))
  return list(zip(*pts))


def _update_pos(curr_pos, end_pos, absolute):
  if absolute:
    return end_pos
  return curr_pos[0] + end_pos[0], curr_pos[1] + end_pos[1]


def constant_color(*unused_args):
  return np.array([255, 255, 255])


def _render_cubic(canvas, curr_pos, c_args, absolute, color):
  """Renders a cubic bezier curve in the given canvas."""
  if not absolute:
    c_args[0] += curr_pos[0]
    c_args[1] += curr_pos[1]
    c_args[2] += curr_pos[0]
    c_args[3] += curr_pos[1]
    c_args[4] += curr_pos[0]
    c_args[5] += curr_pos[1]
  x, y = _cubicbezier(curr_pos[0], curr_pos[1],
                      c_args[0], c_args[1],
                      c_args[2], c_args[3],
                      c_args[4], c_args[5])
  max_possible = len(canvas)
  x = [int(round(x_)) for x_ in x]
  y = [int(round(y_)) for y_ in y]
  within_range = lambda x: 0 <= x < max_possible
  filtered = [(x_, y_) for x_, y_ in zip(x, y)
              if within_range(x_) and within_range(y_)]
  if not filtered:
    return
  x, y = list(zip(*filtered))
  canvas[y, x, :] = color


def _render_line(canvas, curr_pos, l_args, absolute, color):
  """Renders a line in the given canvas."""
  end_point = l_args
  if not absolute:
    end_point[0] += curr_pos[0]
    end_point[1] += curr_pos[1]
  rr, cc, val = draw.line_aa(int(curr_pos[0]), int(curr_pos[1]),
                             int(end_point[0]), int(end_point[1]))

  max_possible = len(canvas)
  within_range = lambda x: 0 <= x < max_possible
  filtered = [(x, y, v) for x, y, v in zip(rr, cc, val)
              if within_range(x) and within_range(y)]
  if not filtered:
    return
  rr, cc, val = list(zip(*filtered))
  val = [(v * color) for v in val]
  canvas[cc, rr, :] = val


def per_step_render(path, absolute=False, color=constant_color):
  """Render the icon's edges, given its path."""
  to_canvas_size = lambda l: [float(f) * (64./24.) for f in l]

  canvas = np.zeros((64, 64, 3))
  curr_pos = (0.0, 0.0)
  for i, cmd in enumerate(path):
    if not cmd: continue
    if cmd[0] in 'mM':
      curr_pos = _update_pos(curr_pos, to_canvas_size(cmd[-2:]), absolute)
    elif cmd[0] in 'cC':
      _render_cubic(canvas, curr_pos, to_canvas_size(cmd[1:]), absolute,
                    color(i, 55))
      curr_pos = _update_pos(curr_pos, to_canvas_size(cmd[-2:]), absolute)
    elif cmd[0] in 'lL':
      _render_line(canvas, curr_pos, to_canvas_size(cmd[1:]), absolute,
                   color(i, 55))
      curr_pos = _update_pos(curr_pos, to_canvas_size(cmd[1:]), absolute)

  return canvas


def zoom_out(path_list, add_baseline=0., per=22):
  """Makes glyph slightly smaller in viewbox, makes some descenders visible."""
  # assumes tensor is already unnormalized, and in long form
  new_path = []
  for command in path_list:
    args = []
    is_even = False
    for arg in command[1:]:
      if is_even:
        args.append(str(float(arg) - ((24.-per) / 24.)*64./4.))
        is_even = False
      else:
        args.append(str(float(arg) - add_baseline))
        is_even = True
    new_path.append([command[0]] + args)
  return new_path


##################### UTILS FOR PROCESSING VECTORS ################
def append_eos(sample, categorical, feature_dim):
  if not categorical:
    eos = -1 * np.ones(feature_dim)
  else:
    eos = np.zeros(feature_dim)
    eos[0] = 1.0
  sample.append(eos)
  return sample


def make_simple_cmds_long(out):
  """Converts svg decoder output to format required by some render functions."""
  # out has 10 dims
  # the first 4 are respectively dims 0, 4, 5, 9 of the full 20-dim onehot vec
  # the latter 6 are the 6 last dims of the 10-dim arg vec
  shape_minus_dim = list(np.shape(out))[:-1]
  return np.concatenate([out[..., :1],
                         np.zeros(shape_minus_dim + [3]),
                         out[..., 1:3],
                         np.zeros(shape_minus_dim + [3]),
                         out[..., 3:4],
                         np.zeros(shape_minus_dim + [14]),
                         out[..., 4:]], -1)


################# UTILS FOR CONVERTING VECTORS TO SVGS ########################
def vector_to_svg(vectors, stop_at_eos=False, categorical=False):
  """Tranforms a given vector to an svg string."""
  new_path = []
  for vector in vectors:
    if stop_at_eos:
      if categorical:
        try:
          is_eos = np.argmax(vector[:len(CMDS_LIST) + 1]) == 0
        except:
          raise Exception(vector)
      else:
        is_eos = vector[0] < -0.5

      if is_eos:
        break
    new_path.append(' '.join(vector_to_cmd(vector, categorical=categorical)))
  new_path = ' '.join(new_path)
  return SVG_PREFIX_BIG + PATH_PREFIX_1 + new_path + PATH_POSFIX_1 + SVG_POSFIX


def vector_to_cmd(vector, categorical=False, return_floats=False):
  """Does the inverse transformation as cmd_to_vector()."""
  cast_fn = float if return_floats else str
  if categorical:
    command = (vector[:len(CMDS_LIST) + 1],)
    arguments = vector[len(CMDS_LIST) + 1:]
    cmd_idx = np.argmax(command) - 1
  else:
    command, arguments = vector[:1], vector[1:]
    cmd_idx = int(round(command[0]))

  if cmd_idx < -0.5:
    # EOS
    return []
  if cmd_idx >= len(CMDS_LIST):
    cmd_idx = len(CMDS_LIST) - 1

  cmd = CMDS_LIST[cmd_idx]
  cmd_list = [cmd]

  if cmd in 'hH':
    cmd_list.append(cast_fn(arguments[8]))  # x
  elif cmd in 'vV':
    cmd_list.append(cast_fn(arguments[9]))  # y
  elif cmd in 'mMlLtT':
    cmd_list.append(cast_fn(arguments[8]))  # x
    cmd_list.append(cast_fn(arguments[9]))  # y
  elif cmd in 'sSqQ':
    cmd_list.append(cast_fn(arguments[6]))  # x2
    cmd_list.append(cast_fn(arguments[7]))  # y2
    cmd_list.append(cast_fn(arguments[8]))  # x
    cmd_list.append(cast_fn(arguments[9]))  # y
  elif cmd in 'cC':
    cmd_list.append(cast_fn(arguments[4]))  # x1
    cmd_list.append(cast_fn(arguments[5]))  # y1
    cmd_list.append(cast_fn(arguments[6]))  # x2
    cmd_list.append(cast_fn(arguments[7]))  # y2
    cmd_list.append(cast_fn(arguments[8]))  # x
    cmd_list.append(cast_fn(arguments[9]))  # y
  elif cmd in 'aA':
    cmd_list.append(cast_fn(arguments[0]))  # rx
    cmd_list.append(cast_fn(arguments[1]))  # ry
    # x-axis-rotation is always 0
    cmd_list.append(cast_fn('0'))
    # the following two flags are binary.
    cmd_list.append(cast_fn(1 if arguments[2] > 0.5 else 0))  # large-arc-flag
    cmd_list.append(cast_fn(1 if arguments[3] > 0.5 else 0))  # sweep-flag
    cmd_list.append(cast_fn(arguments[8]))  # x
    cmd_list.append(cast_fn(arguments[9]))  # y

  return cmd_list


############## UTILS FOR CONVERTING SVGS/VECTORS TO IMAGES ###################
def create_image_conversion_fn(max_outputs, categorical=False):
  """Binds the number of outputs to the image conversion fn (to svg or png)."""
  def convert_to_svg(decoder_output):
    converted = []
    for example in decoder_output:
      if len(converted) == max_outputs:
        break
      converted.append(vector_to_svg(example, True, categorical=categorical))
    return np.array(converted)

  return convert_to_svg


################### UTILS FOR CREATING TF SUMMARIES ##########################
def _make_encoded_image(img_tensor):
  pil_img = Image.fromarray(np.squeeze(img_tensor * 255).astype(np.uint8),
                            mode='L')
  buff = io.BytesIO()
  pil_img.save(buff, format='png')
  encoded_image = buff.getvalue()
  return encoded_image


def make_text_summary_value(svg, tag):
  """Converts the given str to a text tf.summary.Summary.Value."""
  svg_proto = tf.make_tensor_proto(svg, tf.string)
  value = tf.summary.Summary.Value(tag=tag, tensor=svg_proto)
  value.metadata.plugin_data.plugin_name = 'text'
  return value


def make_image_summary(image_tensor, tag):
  """Converts the given image tensor to a tf.summary.Summary.Image."""
  encoded_image = _make_encoded_image(image_tensor)
  image_sum = tf.summary.Summary.Image(encoded_image_string=encoded_image)
  return tf.summary.Summary.Value(tag=tag, image=image_sum)
