
import inspect
from scipy.signal.windows import gaussian
import numpy as np

# Use as Zcorr = Z * np.exp(1j * freqs * config.PHASE_CORR).
# Alors, should be 2*pi*time_of_flight = 1.70e-6 (roughly the same),
# except for additional phase slope from microwave environment.
PHASE_CORR = 1.6239384827049928e-06 + 3.437388002238489e-08
VNA_PHASE_CORR = PHASE_CORR

## Note:
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

## OPX analog input gain
input_gain  = +15 # OPX IN, range -12 to +20dB

## Octave output parameters
resonator_output_gain = -10 # Octave RF1, range -20 to +20dB
vna_output_gain = -10 # Octave RF1, range -20 to +20dB
qubit_output_gain = -10 # Octave RF2, range -20 to +20dB

## ADC offset in Volts
adcoffset = np.array([0.0572889, 0.0706217])

## Frequencies
resonatorLO = 5.010e9
resonatorIF = 0.20146e9
# 20ns readout -> 200MHz IF
# 50ns readout -> 100MHz IF

qubitLO = 5.3e9
qubitIF = -150e6#-380e6

vnaLO = 5.31e9
vnaIF = -0.100e9

# For resonator width 450kHz width, ie. t=2us lifetime
# choose at least 6us=3t, ie. 1500cycles
cooldown_clk = 1000 # 4ns cycles

## Readout pulse parameters
const_len = 10000
const_amp = 0.316

readout_len = 100000 # ns
readout_amp = 0.01 # -30dB

short_readout_len = 20000 # ns
short_readout_amp = 0.0316
short_readout_amp_gain = 0#+20 # dB added to resonator output gain

long_readout_len = 7000
long_readout_amp = 0.1

# Time it takes the pulses to go through the RF chain, including the device.
# In ns, minimum 24, multiple of 4
time_of_flight = 24 + 252

## Qubit parameters:

# Needs to be several T1 so that the final state is an equal population of |0> and |1>
# Should be longer than readout_len for some protocols
saturation_len = 21000#21000#500 #16 # ns
saturation_amp = 0.01 # 0.0177 #0.316

preload_len = 16 # ns
preload_amp = 0.5
preload_wf = np.zeros(max(16, preload_len))
preload_wf[-preload_len:] = preload_amp

# Pi pulse parameters
pi_len = 100  # in units of ns
pi_amp = 0.149  # in units of volts
pi_wf = (pi_amp * (gaussian(pi_len, pi_len / 5) -
                   gaussian(pi_len, pi_len / 5)[-1])).tolist()  # waveform
minus_pi_wf = ((-1) * pi_amp * (gaussian(pi_len, pi_len / 5) -
                   gaussian(pi_len, pi_len / 5)[-1])).tolist()  # waveform


# Pi_half pulse parameters
pi_half_len = 100  # in units of ns
pi_half_amp = 0.5 * pi_amp  # in units of volts
pi_half_wf = (pi_half_amp * (gaussian(pi_half_len, pi_half_len / 5) -
                             gaussian(pi_half_len, pi_half_len / 5)[-1])).tolist()  # waveform
minus_pi_half_wf = ((-1) * pi_half_amp * (gaussian(pi_half_len, pi_half_len / 5) -
                             gaussian(pi_half_len, pi_half_len / 5)[-1])).tolist()  # waveform


# Gaussian pulse parameters
# The gaussian is used when calibrating pi and pi_half pulses
gauss_len = const_len
gauss_amp = const_amp
gauss_wf = (gauss_amp * (gaussian(gauss_len, gauss_len / 5) -
                         gaussian(gauss_len, gauss_len / 5)[-1])).tolist()  # waveform

# Flux:
square_flux_amp = 0.3
triangle_flux_amp = 0.3
triangle_wf = [triangle_flux_amp * i/7 for i in range(8)] + [triangle_flux_amp * (1 - i/7) for i in range(8)]

# Rotation angle:
rotation_angle = (-8.0/180) * np.pi  # angle in degrees

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
                1: {'offset': 0.0},  # I vna probe
                2: {'offset': 0.0},  # Q vna probe
                3: {'offset': 0.0},  # I qubit drive
                4: {'offset': 0.0},  # Q qubit drive
                7: {'offset': 0.0},  # I resonator
                8: {'offset': 0.0},  # Q resonator

            },

            'digital_outputs': {},

            'analog_inputs': {
                # Gain range -12 to +20dB
                1: {'offset': adcoffset[0], 'gain_db': input_gain},  # I from down conversion
                2: {'offset': adcoffset[1], 'gain_db': input_gain},  # Q from down conversion
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
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi_half': 'pi_half_pulse',
                'X': 'Xpi_pulse',
                '-X': '-Xpi_pulse',
                'X/2': 'Xpi_half_pulse',
                '-X/2': '-Xpi_half_pulse',
                'Y': 'Ypi_pulse',
                '-Y': '-Ypi_pulse',
                'Y/2': 'Ypi_half_pulse',
                '-Y/2': '-Ypi_half_pulse',
            },
        },
        
        'vna': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': vnaLO,
                'mixer': 'octave_octave1_1',
            },
            'intermediate_frequency': vnaIF,
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

        'gaussian_pulse': {
            'operation': 'control',
            'length': gauss_len,  # in ns
            'waveforms': {
                'I': 'gaussian_wf',
                'Q': 'zero_wf'
            }
        },

        'pi_pulse': {
            'operation': 'control',
            'length': pi_len,  # in ns
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf',
            },
        },

        'pi_half_pulse': {
            'operation': 'control',
            'length': pi_half_len,  # in ns
            'waveforms': {
                'I': 'pi_half_wf',
                'Q': 'zero_wf',
            },
        },

        'Xpi_pulse': {
            'operation': 'control',
            'length': pi_len,  # in ns
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf',
            },
        },

        '-Xpi_pulse': {
            'operation': 'control',
            'length': pi_len,  # in ns
            'waveforms': {
                'I': '-pi_wf',
                'Q': 'zero_wf',
            },
        },

        'Xpi_half_pulse': {
            'operation': 'control',
            'length': pi_half_len,  # in ns
            'waveforms': {
                'I': 'pi_half_wf',
                'Q': 'zero_wf',
            },
        },

        '-Xpi_half_pulse': {
            'operation': 'control',
            'length': pi_half_len,  # in ns
            'waveforms': {
                'I': '-pi_half_wf',
                'Q': 'zero_wf',
            },
        },

        'Ypi_pulse': {
            'operation': 'control',
            'length': pi_len,  # in ns
            'waveforms': {
                'I': 'zero_wf',
                'Q': 'pi_wf',
            },
        },

        '-Ypi_pulse': {
            'operation': 'control',
            'length': pi_len,  # in ns
            'waveforms': {
                'I': 'zero_wf',
                'Q': '-pi_wf',
            },
        },

        'Ypi_half_pulse': {
            'operation': 'control',
            'length': pi_half_len,  # in ns
            'waveforms': {
                'I': 'zero_wf',
                'Q': 'pi_half_wf',
            },
        },

        '-Ypi_half_pulse': {
            'operation': 'control',
            'length': pi_half_len,  # in ns
            'waveforms': {
                'I': 'zero_wf',
                'Q': '-pi_half_wf',
            },
        },

        'square_pulse': {
            "operation": "control",
            "length": 16,  # in ns
            "waveforms": {"single": "square_wf"}
        },

        'triangle_pulse': {
            "operation": "control",
            "length": 16,  # in ns
            "waveforms": {"single": "triangle_wf"}
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

        'long_readout_pulse': {
            "operation": "measurement",
            "length": long_readout_len,  # in ns
            'waveforms': {
                'I': 'long_readout_wf',
                'Q': 'zero_wf'
            },
            "digital_marker": "ON",
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
        'square_wf': {"type": "constant", "sample": square_flux_amp},
        'triangle_wf': {"type": "arbitrary", "samples": triangle_wf},
        'gaussian_wf': {"type": "arbitrary", "samples": gauss_wf},
        'pi_wf': {'type': 'arbitrary', 'samples': pi_wf},
        '-pi_wf': {'type': 'arbitrary', 'samples': minus_pi_wf},
        'pi_half_wf': {'type': 'arbitrary', 'samples': pi_half_wf},
        '-pi_half_wf': {'type': 'arbitrary', 'samples': minus_pi_half_wf},
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
            
            "octave_octave1_1": [
                {
                    "intermediate_frequency": vnaIF,
                    "lo_frequency": vnaLO,
                    "correction": (0., 0., 0., 0.),
                },
            ],
            
            "octave_octave1_2": [
                {
                    "intermediate_frequency": qubitIF,
                    "lo_frequency": qubitLO,
                    "correction": IQ_imbalance(-0.05,0.10),
                },
            ],
    }
}

# Expose content of this module as object to be included in experiment metadata
meta = {item: globals()[item] for item in dir() if not (
    item.startswith("__") or item == 'meta'
    or inspect.ismodule(globals()[item])
    or hasattr(globals()[item], '__call__'))}
