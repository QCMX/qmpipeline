
import importlib
import qm.qua as qua
import qm.octave as octave

import configuration as config
#import configuration_RF1 as config
import qminit

#qmm = qminit.connect(use_calibration=False)
qmm = qminit.connect()

#%%
# Simply output constant pulse

importlib.reload(config)
importlib.reload(qminit)

with qua.program() as mixer_cal_resonator:
    #qua.update_frequency('resonator', config.resonator_if)
    with qua.infinite_loop_():
        qua.play('const', 'resonator')
        #qua.play('const', 'qubit')

# Use in real time to correct offsets:
# qm.set_output_dc_offset_by_element('resonator', 'Q', 0.015)
# Use in real time to correct IQ imbalance:
# job.set_element_correction('qubit', IQ_imbalance(-0.08,-0.03))

# Used to correct for IQ mixer imbalances
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]

# execute QUA program
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
job = qm.execute(mixer_cal_resonator)

#%%
# Run calibration, then output constant pulse
# Calibration closes all running qm's

importlib.reload(config)
importlib.reload(qminit)

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)

print("Running calibration on resonator channel...")
qm.octave.calibrate_element('resonator', [
    (config.resonatorLO, 10e6),
    (config.resonatorLO, 50e6),
    (config.resonatorLO, 100e6),
    (config.resonatorLO, 150e6),
    (config.resonatorLO, 200e6),
    (config.resonatorLO, 250e6),
    (config.resonatorLO, 300e6),
    (config.resonatorLO, 350e6),
    #(config.resonatorLO, config.resonatorIF)
    ])

# Need to reopen qm to apply calibration settings
print("Playing constant pulse on resonator channel...")
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
job = qm.execute(mixer_cal_resonator)

#%%

importlib.reload(config)
importlib.reload(qminit)

with qua.program() as mixer_cal_qubit:
    qua.update_frequency('qubit', -300e6)
    with qua.infinite_loop_():
        qua.play('const', 'qubit')

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_qubit(qm, config)
job = qm.execute(mixer_cal_qubit)

#qm.set_output_dc_offset_by_element('qubit', 'Q', 0.015)

#%%

importlib.reload(config)
importlib.reload(qminit)

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_qubit(qm, config)

print("Running calibration on qubit channel...")
cal = qm.octave.calibrate_element('qubit', [
    (config.qubitLO, 10e6),
    (config.qubitLO, 50e6),
    (config.qubitLO, 100e6),
    (config.qubitLO, 150e6),
    (config.qubitLO, 200e6),
    (config.qubitLO, 250e6),
    (config.qubitLO, 300e6),
    (config.qubitLO, 350e6),
    (config.qubitLO, config.qubitIF)
    ])

# Need to reopen qm to apply calibration settings
print("Playing constant pulse on qubit channel...")
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_qubit(qm, config)
job = qm.execute(mixer_cal_qubit)

#%%

job.halt()
qm.octave.set_rf_output_mode('resonator', octave.RFOutputMode.off)
qm.octave.set_rf_output_mode('qubit', octave.RFOutputMode.off)

#%%

import json
with open('calibration_db.json') as f:
    cj = json.load(f)
print(json.dumps(cj, indent=4))

#%%

import numpy as np

# Used to correct for IQ mixer imbalances
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]
