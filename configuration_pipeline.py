
import inspect
from scipy.signal.windows import gaussian
import numpy as np

# Use as Zcorr = Z * np.exp(1j * freqs * config.PHASE_CORR).
# Alors, should be 2*pi*time_of_flight = 1.70e-6 (roughly the same),
# except for additional phase slope from microwave environment.
PHASE_CORR = 1.6239384827049928e-06

# Note:
# OPX outputs have 16bits, ie. 1V/2**16 =  15uV resolution
# OPX inputs  have 12bits, ie. 1V/2**12 = 244uV resolution

# OPX output power
#  0.5V =  +4dBm
# 316mV =   0dBm
# 177mV =  -5dBm
# 100mV = -10dBm
#  56mv = -15dBm
#  31.6mV = -20dBm
#  17.8mV = -25dBm
#  10.0mV = -30dBm
#   3.16mV = -40dBm

# Octave parameters
resonator_output_gain = -20 # Octave RF1, range -20 to +20dB
resonator_input_gain  = +10 # OPX IN, range -12 to +20dB
qubit_output_gain     = -10 # Octave RF2, range -20 to +20dB
# Qubit output gain is limited by 2*drive LO and LO + IF also going through
# the cavity and saturating the readout.

# ADC offset in Volts
adcoffset = np.array([0.057299890319824215 , 0.07062872677001952])

## Frequencies
resonatorLO = 5.010e9
resonatorIF = 0.20146e9
# 20ns readout -> 200MHz IF
# 50ns readout -> 100MHz IF

qubitLO = 3.5e9
qubitIF = -93e6

# 4000 cycles = 16us
cooldown_clk = 4000 # cycles

## Readout pulse parameters
const_len = 10000
const_amp = 0.316

readout_len = 20000 # ns
readout_amp = 0.0056

short_readout_len = 1000 # ns
short_readout_amp = 0.316

# Time it takes the pulses to go through the RF chain, including the device.
# In ns, minimum 24, multiple of 4
time_of_flight = 24 + 252

## Qubit parameters:

# Needs to be several T1 so that the final state is an equal population of |0> and |1>
# Should be longer than readout_len for some protocols
saturation_len = 21000 # ns
saturation_amp = 0.316

# Pi pulse parameters
pi_len = 100  # in units of ns
pi_amp = 0.149  # in units of volts


# Rotation angle for rotated readout demodulation
rotation_angle = (-0/180) * np.pi  # angle in degrees

# Used to correct for IQ mixer imbalances
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


qmconfig = {

    'version': 1,

    'controllers': {

        'con1': {

            'type': 'opx1',

            'analog_outputs': {
                3: {'offset': 0.0},  # I resonator
                4: {'offset': 0.0},  # Q resonator
                5: {'offset': 0.0},  # Q resonator
                7: {'offset': -0.03259},  # I resonator
                8: {'offset': -0.04925},  # Q resonator
            },

            'digital_outputs': {},

            'analog_inputs': {
                # Gain range -12 to +20dB
                1: {'offset': adcoffset[0], 'gain_db': resonator_input_gain},  # I from down conversion
                2: {'offset': adcoffset[1], 'gain_db': resonator_input_gain},  # Q from down conversion
            },
        },
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': qubitLO,
                'mixer': 'octave_octave1_2',
            },
            'intermediate_frequency': qubitIF,
            'operations': {
                'const': 'const_pulse',
                'saturation': 'saturation_pulse',
            },
        },

        # second qubit on same OPX & octave output to multiplex pulses
        # eg. for three-tone spectroscopy
        'qubit2': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': qubitLO,
                'mixer': 'octave_octave1_2',
            },
            'intermediate_frequency': qubitIF,
            'operations': {
                'const': 'const_pulse',
                'saturation': 'saturation_pulse',
            },
        },

        'resonator': {
            'mixInputs': {
                'I': ('con1', 7),
                'Q': ('con1', 8),
                'lo_frequency': resonatorLO,
                'mixer': 'octave_octave1_4'
            },
            'intermediate_frequency': resonatorIF,
            'operations': {
                'const': 'const_pulse',
                'short_readout': 'short_readout_pulse',
                'readout': 'readout_pulse',
            },
            'time_of_flight': time_of_flight,
            'smearing': 0, # max 127
            'outputs': {
                'out1': ('con1', 1),
                'out2': ('con1', 2),
            },
        },
    },

    'pulses': {

        'const_pulse': {
            'operation': 'control',
            'length': const_len,  # in ns
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf',
            },
        },

        'saturation_pulse': {
            'operation': 'control',
            'length': saturation_len,  # in ns
            'waveforms': {
                'I': 'saturation_wf',
                'Q': 'zero_wf'
            }
        },

        'short_readout_pulse': {
            "operation": "measurement",
            "length": short_readout_len,  # in ns
            'waveforms': {
                'I': 'short_readout_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON',
            'integration_weights': {
                'cos': 'short_cos_weights',
                'sin': 'short_sin_weights',
                'minus_sin': 'short_minus_sin_weights',
                'rotated_cos': 'short_rotated_cos_weights',
                'rotated_sin': 'short_rotated_sin_weights',
                'rotated_minus_sin': 'short_rotated_minus_sin_weights'
            },
        },

        'readout_pulse': {
            "operation": "measurement",
            "length": readout_len,  # in ns
            'waveforms': {
                'I': 'readout_wf',
                'Q': 'zero_wf'
            },
            "digital_marker": "ON",
            "integration_weights": {
                'cos': 'cos_weights',
                'sin': 'sin_weights',
                'minus_sin': 'minus_sin_weights',
                'rotated_cos': 'rotated_cos_weights',
                'rotated_sin': 'rotated_sin_weights',
                'rotated_minus_sin': 'rotated_minus_sin_weights'
            },
        },
    },

    'waveforms': {
        'const_wf': {'type': 'constant', 'sample': const_amp},
        'zero_wf': {"type": "constant", "sample": 0.0},
        'saturation_wf': {"type": "constant", "sample": saturation_amp},
        'short_readout_wf': {"type": "constant", "sample": short_readout_amp},
        'readout_wf': {"type": "constant", "sample": readout_amp},
    },

    'digital_waveforms': {

        "ON": {"samples": [(1, 0)]},  # commonly used for measurement pulses, e.g., in a readout pulse

    },

    'integration_weights': {

        'short_cos_weights': {
            "cosine": [(1.0, short_readout_len)],  # Previous format for versions before 1.20: [1.0] * readout_len
            "sine": [(0.0, short_readout_len)],
        },

        'short_sin_weights': {
            "cosine": [(0.0, short_readout_len)],
            "sine": [(1.0, short_readout_len)],
        },

        'short_minus_sin_weights': {
            "cosine": [(0.0, short_readout_len)],
            "sine": [(-1.0, short_readout_len)],
        },

        'short_rotated_cos_weights': {
            "cosine": [(np.cos(rotation_angle), short_readout_len)],
            "sine": [(-np.sin(rotation_angle), short_readout_len)],
        },

        'short_rotated_sin_weights': {
            "cosine": [(np.sin(rotation_angle), short_readout_len)],
            "sine": [(np.cos(rotation_angle), short_readout_len)],
        },

        'short_rotated_minus_sin_weights': {
            "cosine": [(-np.sin(rotation_angle), short_readout_len)],
            "sine": [(-np.cos(rotation_angle), short_readout_len)],
        },

        'cos_weights': {
            "cosine": [(1.0, readout_len)],  # Previous format for versions before 1.20: [1.0] * readout_len
            "sine": [(0.0, readout_len)],
        },

        'sin_weights': {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },

        'minus_sin_weights': {
            "cosine": [(0.0, readout_len)],
            "sine": [(-1.0, readout_len)],
        },

        'rotated_cos_weights': {
            "cosine": [(np.cos(rotation_angle), readout_len)],
            "sine": [(-np.sin(rotation_angle), readout_len)],
        },

        'rotated_sin_weights': {
            "cosine": [(np.sin(rotation_angle), readout_len)],
            "sine": [(np.cos(rotation_angle), readout_len)],
        },

        'rotated_minus_sin_weights': {
            "cosine": [(-np.sin(rotation_angle), readout_len)],
            "sine": [(-np.cos(rotation_angle), readout_len)],
        },
    },

    'mixers': {
        "octave_octave1_4": [
            {
                "intermediate_frequency": resonatorIF,
                "lo_frequency": resonatorLO,
                "correction": (1.0536929927766323, 0.18822958320379257, 0.18970587104558945, 1.045493170619011),
            },
        ],

        "octave_octave1_2": [
            {
                "intermediate_frequency": qubitIF,
                "lo_frequency": qubitLO,
                "correction": (1, 0, 0, 1),
            },
        ],
    },
}

# Expose content of this module as object to be included in experiment metadata
meta = {item: globals()[item] for item in dir() if not (
    item.startswith("__") or item == 'meta'
    or inspect.ismodule(globals()[item])
    or hasattr(globals()[item], '__call__'))}
