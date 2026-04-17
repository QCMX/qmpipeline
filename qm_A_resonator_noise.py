
import numpy as np
import matplotlib.pyplot as plt

import qmtools
import qminit
from helpers import mpl_pause

qmm = qminit.connect()

#%%

import importlib
import configuration_pipeline as config_pipeline
# import configuration as config
importlib.reload(config_pipeline)
localconfig = qmtools.config_module_to_dict(config_pipeline)

qmtools.QMMixerCalibration(qmm, localconfig).run()

qmtools.QMTimeOfFlight(qmm, localconfig, Navg=100).run()

#%%

spec = qmtools.QMResonatorSpec(
    qmm, localconfig, Navg=30,
    resonatorIFs=np.arange(202e6, 222e6, 0.05e6))
results = spec.run(plot=True)
resonatorfit = spec.fit_lorentzian(ax=spec.ax)
localconfig['resonatorIF'] = resonatorfit[0][0]

#%%

noiseprog = qmtools.QMNoiseSpectrum(qmm, localconfig, Nsamples=100000, wait_ns=16)
qmprog = noiseprog._make_program()

qm = qmm.open_qm(localconfig['qmconfig'])
noiseprog._init_octave(qm)
jobid = qm.compile(qmprog)

fig, ax = plt.subplots()
line = None

while True:
    job = qm.queue.add_compiled(jobid).wait_for_execution()
    handles = job.result_handles
    handles.wait_for_all_values()
    result = noiseprog._retrieve_results(handles)

    if line is None:
        line, = ax.plot(result['fftfreq'], np.abs(result['fft']))
    else:
        line.set_ydata(np.abs(result['fft']))
    mpl_pause(0.2)
