
# Note:
# Used calibration values are the chosen from the database when opening the QM
# based upon the element's/mixer's LO and IF.
# The element and mixer LO and IF have to match, otherwise an error is produced.

# While the minimum LO is 2GHz, the calibration works reliable for negative IF
# with total frequency below 2GHz.

# The calibration is as similar for positive/negative IF as for
# very different positive IF.

import importlib
import qm.qua as qua
import qm.octave as octave

import configuration_novna as config
#import configuration_RF1 as config
import qminit

#qmm = qminit.connect(use_calibration=False)
qmm = qminit.connect()
from qualang_tools.loops import from_array

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
    (config.resonatorLO, config.resonatorIF)
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
    #qua.update_frequency('qubit', -385e6)
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
cal = qm.octave.calibrate_element('qubit', [(config.qubitLO, config.qubitIF)])

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

qms = qmm.list_open_quantum_machines()
assert len(qms) <= 1
if qms:
    print("Close", qm)
    qm = qmm.get_qm(qms[0])
    qm.close()
else:
    print("No QM open")

#%%
import numpy as np

fs = np.arange(-450e6, +450e6, 1e6)
with qua.program() as ifview_qubit:
    n = qua.declare(int)
    f = qua.declare(int)
    with qua.infinite_loop_():
        with qua.for_(*from_array(f, fs)):
            qua.update_frequency('qubit', f)
            with qua.for_(n, 0, n < 1000, n + 1):
                qua.play('const', 'qubit')

job = qm.execute(ifview_qubit)

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

#%%
#Check calibration for a list of LO frequencies

import numpy as np

qubitLOs = [2e9, 2.1e9, 2.2e9, 2.3e9, 2.4e9, 2.45e9, 2.6e9, 2.7e9, 2.8e9, 2.9e9, 3e9, 3.1e9, 3.2e9, 3.3e9, 3.4e9, 3.5e9, 3.6e9, 3.7e9]
for qubitLO in qubitLOs:
    print(qubitLO/1e9, "GHz")
    config.qubitLO = qubitLO
    config.qmconfig['elements']['qubit']['mixInputs']['lo_frequency'] = qubitLO
    config.qmconfig['mixers']['octave_octave1_2'][0]['lo_frequency'] = qubitLO
    fs = np.arange(-450e6, +450e6, 1e6)
    with qua.program() as ifview_qubit:
        n = qua.declare(int)
        f = qua.declare(int)
        with qua.infinite_loop_():
            with qua.for_(*from_array(f, fs)):
                qua.update_frequency('qubit', -50e6)
                with qua.for_(n, 0, n < 1000, n + 1):
                    qua.play('const', 'qubit')

    qm = qmm.open_qm(config.qmconfig)
    qminit.octave_setup_qubit(qm, config)
    job = qm.execute(ifview_qubit)
    input("Press enter for next LO freq")
