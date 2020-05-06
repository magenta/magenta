import collections
from enum import Enum


class Family(Enum):
    BASS = 0
    BRASS = 1
    FLUTE = 2
    GUITAR = 3
    KEYBOARD = 4
    MALLET = 5
    ORGAN = 6
    REED = 7
    STRING = 8
    SYNTH_LEAD = 9
    VOCAL = 10
    OTHER = 11
    DRUMS = 12


midi_instrument_to_family = collections.defaultdict(lambda: Family.OTHER)
midi_instrument_to_family.update({
    0: Family.KEYBOARD,
    1: Family.KEYBOARD,
    2: Family.KEYBOARD,
    3: Family.KEYBOARD,
    4: Family.KEYBOARD,
    5: Family.KEYBOARD,
    6: Family.KEYBOARD,
    7: Family.KEYBOARD,
    8: Family.MALLET,
    9: Family.MALLET,
    10: Family.MALLET,
    11: Family.MALLET,
    12: Family.MALLET,
    13: Family.MALLET,
    14: Family.MALLET,
    15: Family.MALLET,
    16: Family.ORGAN,
    17: Family.ORGAN,
    18: Family.ORGAN,
    19: Family.ORGAN,
    20: Family.ORGAN,
    21: Family.ORGAN,
    22: Family.ORGAN,
    23: Family.ORGAN,
    24: Family.GUITAR,
    25: Family.GUITAR,
    26: Family.GUITAR,
    27: Family.GUITAR,
    28: Family.GUITAR,
    29: Family.GUITAR,
    30: Family.GUITAR,
    31: Family.GUITAR,
    32: Family.BASS,
    33: Family.BASS,
    34: Family.BASS,
    35: Family.BASS,
    36: Family.BASS,
    37: Family.BASS,
    38: Family.BASS,
    39: Family.BASS,
    40: Family.STRING,
    41: Family.STRING,
    42: Family.STRING,
    43: Family.STRING,
    44: Family.STRING,
    45: Family.STRING,
    46: Family.STRING,
    47: Family.STRING,  # TIMPANI?
    48: Family.STRING,
    49: Family.STRING,
    50: Family.STRING,
    51: Family.STRING,
    52: Family.VOCAL,
    53: Family.VOCAL,
    54: Family.VOCAL,
    55: Family.STRING,  # orch hit
    56: Family.BRASS,
    57: Family.BRASS,
    58: Family.BRASS,
    59: Family.BRASS,
    60: Family.BRASS,
    61: Family.BRASS,
    62: Family.BRASS,
    63: Family.BRASS,
    64: Family.REED,
    65: Family.REED,
    66: Family.REED,
    67: Family.REED,
    68: Family.REED,
    69: Family.REED,
    70: Family.REED,
    71: Family.REED,
    72: Family.FLUTE,
    73: Family.FLUTE,
    74: Family.FLUTE,
    75: Family.FLUTE,
    76: Family.FLUTE,
    77: Family.FLUTE,
    78: Family.FLUTE,
    79: Family.FLUTE,
    80: Family.SYNTH_LEAD,
    81: Family.SYNTH_LEAD,
    82: Family.SYNTH_LEAD,
    83: Family.SYNTH_LEAD,
    84: Family.SYNTH_LEAD,
    85: Family.VOCAL,
    86: Family.SYNTH_LEAD,
    87: Family.SYNTH_LEAD,
    105: Family.GUITAR,
    106: Family.GUITAR,
    107: Family.GUITAR,
    108: Family.GUITAR,
    109: Family.MALLET,
    110: Family.REED,
    111: Family.STRING,
    112: Family.REED,
    113: Family.MALLET,
    114: Family.MALLET,
})

family_to_midi_instrument = {
    0: 33,  # Acoustic Bass
    1: 57,  # Trumpet
    2: 74,  # Flute
    3: 25,  # Acoustic Nylon Guitar
    4: 1,  # keyboard / Acoustic Grand Piano
    5: 9,  # mallet / Celesta
    6: 17,  # organ / Drawbar Organ
    7: 66,  # reed / Alto Sax
    8: 49,  # string / String Ensemble
    9: 83,  # synth lead / Square
    10: 54,  # vocal / Voice Oohs
    11: 118,
    12: 119, # TODO actual percussion
}