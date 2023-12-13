
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

# Octave parameters
resonator_output_gain = -20 # Octave RF1, range -20 to +20dB
resonator_input_gain  = +10 # OPX IN, range -12 to +20dB
qubit_output_gain     = 0 # Octave RF2, range -20 to +20dB

# ADC offset in Volts
adcoffset = np.array([0.033352255153200054 , 0.039610600610871925])

## Frequencies
resonatorLO = 4.8500e9
resonatorIF = 0.20715e9
# 20ns readout -> 200MHz IF
# 50ns readout -> 100MHz IF

qubitLO = 2.6e9
qubitIF = 339e6

# For resonator width 450kHz width, ie. t=2us lifetime
# choose at least 6us=3t, ie. 1500cycles
cooldown_clk = 1000 # 4ns cycles

## Readout pulse parameters
const_len = 16
const_amp = 0.1 #0.316

readout_len = 20000 # 52 # ns
readout_amp = 0.316

short_readout_len = 100 # ns
short_readout_amp = 0.0316
short_readout_amp_gain = 20 # dB added to resonator output gain

long_readout_len = 7000
long_readout_amp = 0.1

# Time it takes the pulses to go through the RF chain, including the device.
# In ns, minimum 24, multiple of 4
time_of_flight = 24 + 252

## Qubit parameters:

# Needs to be several T1 so that the final state is an equal population of |0> and |1>
# Should be longer than readout_len for some protocols
saturation_len = 21000 # ns
saturation_amp = 0.316 #316

preload_len = 16 # ns
preload_amp = 0.5
preload_wf = np.zeros(max(16, preload_len))
preload_wf[-preload_len:] = preload_amp

# IQ rotated demodulation: rotation angle:
rotation_angle = (-8.0/180) * np.pi  # angle in degrees

qmconfig = {
    'version': 1,
    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                3: {'offset': 0.0},  # I resonator
                4: {'offset': 0.0},  # Q resonator
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
                'long_readout': 'long_readout_pulse',
                'preload': 'preload_pulse',
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

        'preload_pulse': {
            'operation': 'control',
            'length': len(preload_wf),
            'waveforms': {
                'I': 'preload_wf',
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
            "integration_weights": {
                'cos': 'cos_weights',
                'sin': 'sin_weights',
                'minus_sin': 'minus_sin_weights',
                'rotated_cos': 'rotated_cos_weights',
                'rotated_sin': 'rotated_sin_weights',
                'rotated_minus_sin': 'rotated_minus_sin_weights'
            },
        },

        'long_readout_pulse': {
            "operation": "measurement",
            "length": long_readout_len,  # in ns
            'waveforms': {
                'I': 'long_readout_wf',
                'Q': 'zero_wf'
            },
            "integration_weights": {
                'cos': 'long_cos_weights',
                'sin': 'long_sin_weights',
                'minus_sin': 'long_minus_sin_weights',
                'rotated_cos': 'long_rotated_cos_weights',
                'rotated_sin': 'long_rotated_sin_weights',
                'rotated_minus_sin': 'long_rotated_minus_sin_weights'
            },
        },
    },

    'waveforms': {
        'const_wf': {'type': 'constant', 'sample': const_amp},
        'zero_wf': {"type": "constant", "sample": 0.0},
        'saturation_wf': {"type": "constant", "sample": saturation_amp},
        'preload_wf': {'type': 'arbitrary', 'samples': preload_wf},
        'short_readout_wf': {"type": "constant", "sample": short_readout_amp},
        'readout_wf': {"type": "constant", "sample": readout_amp},
        'long_readout_wf': {"type": "constant", "sample": long_readout_amp},
    },
    'digital_waveforms': {},
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
        'long_cos_weights': {
            "cosine": [(1.0, long_readout_len)],  # Previous format for versions before 1.20: [1.0] * readout_len
            "sine": [(0.0, long_readout_len)],
        },
        'long_sin_weights': {
            "cosine": [(0.0, long_readout_len)],
            "sine": [(1.0, long_readout_len)],
        },
        'long_minus_sin_weights': {
            "cosine": [(0.0, long_readout_len)],
            "sine": [(-1.0, long_readout_len)],
        },
        'long_rotated_cos_weights': {
            "cosine": [(np.cos(rotation_angle), long_readout_len)],
            "sine": [(-np.sin(rotation_angle), long_readout_len)],
        },
        'long_rotated_sin_weights': {
            "cosine": [(np.sin(rotation_angle), long_readout_len)],
            "sine": [(np.cos(rotation_angle), long_readout_len)],
        },
        'long_rotated_minus_sin_weights': {
            "cosine": [(-np.sin(rotation_angle), long_readout_len)],
            "sine": [(-np.cos(rotation_angle), long_readout_len)],
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
                "correction": (1,0,0,1),
            },
        ],
    }
}
