
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# - electrical delay correction (config.PHASE_CORR)
# - resonator IF, readout power

import time
import numpy as np
import matplotlib.pyplot as plt
import importlib
import qm.qua as qua

from helpers import data_path, mpl_pause, plt2dimg, plt2dimg_update
from qm_helpers import int_forloop_values

import configuration as config
import qminit

qmm = qminit.connect()

#%%
importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_qubit_spec_movingdemod'
fpath = data_path(filename, datesuffix='_qm')

Navg = 1000000

drivedelay = 1000//4 # cycles
chunksize = 5 # cycles
chunksperwindow = 1 # 1 is like sliced demod
nwindows = config.short_readout_len // (4 * chunksize)
assert 4 * chunksize * nwindows == config.short_readout_len

try:
    Vgate = gate.get_voltage()
except:
    Vgate = np.nan

with qua.program() as qubit_movingdemod:
    n = qua.declare(int)
    # readout len = 4 * 
    I = qua.declare(qua.fixed, size=nwindows)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed, size=nwindows)  # demodulated and integrated signal
    ind = qua.declare(int)
    n_st = qua.declare_stream()
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Navg, n + 1):
        qua.wait(rand.rand_int(50)+4, 'resonator') # randomize demodulation error
        qua.align()
        qua.wait(drivedelay, 'qubit')
        qua.play('saturation', 'qubit')
        qua.measure('short_readout', 'resonator', None,
                qua.demod.moving_window('cos', I, chunksize, chunksperwindow, 'out1'),
                qua.demod.moving_window('sin', Q, chunksize, chunksperwindow, 'out2'))
        with qua.for_(ind, 0, ind < nwindows, ind + 1):
            qua.save(I[ind], I_st)
            qua.save(Q[ind], Q_st)
        # qua.save(I, I_st)
        # qua.save(Q, Q_st)
        qua.wait(config.cooldown_clk, 'resonator')
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(nwindows).average().save('I')
        Q_st.buffer(nwindows).average().save('Q')

# #%%
# from qm import LoopbackInterface, SimulationConfig
# simulate_config = SimulationConfig(
#     duration=20000, # cycles
#     simulation_interface=LoopbackInterface(([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])))
# job = qmm.simulate(config.qmconfig, qubit_movingdemod, simulate_config)  # do simulation with qmm

# plt.figure()
# job.get_simulated_samples().con1.plot()  # visualize played pulses
# #plt.savefig(fpath+"_pulses.png")
# #%%

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
qminit.octave_setup_qubit(qm, config)

# Execute program
job = qm.execute(qubit_movingdemod)
tstart = time.time()
print("executing")
res_handles = job.result_handles
I_handle = res_handles.get('I')
# I_handle.wait_for_values(1)
Q_handle = res_handles.get('Q')
# Q_handle.wait_for_values(1)
iteration_handle = res_handles.get('iteration')
print("waiting for first value")
iteration_handle.wait_for_values(1)

# Live plotting
t = np.arange(nwindows)*chunksize*4
fig, (ax, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
lineI, = ax.plot(t, np.full(t.size, np.nan))
lineQ, = ax.plot(t, np.full(t.size, np.nan))
ax.set_xlabel('time / ns')
ax.set_ylabel('I,Q')
fig.suptitle('qubit')

try:
    while res_handles.is_processing():
        iteration = iteration_handle.fetch_all() + 1
        I = I_handle.fetch_all() / config.short_readout_len * 2**12
        Q = Q_handle.fetch_all() / config.short_readout_len * 2**12
        lineI.set_ydata(I), lineQ.set_ydata(Q)
        fig.suptitle(f'{iteration}/{Navg}')
        ax.relim()
        ax.autoscale()
        ax.autoscale_view()
        print(f"n={iteration}, remaining: {(Navg-iteration) * (time.time()-tstart)/iteration:.0f}s")
        mpl_pause(0.5)
except Exception as e:
    job.halt()
    raise e
except KeyboardInterrupt:
    job.halt()
print("Execution time", time.time()-tstart, "s")

# Get and save data
Nactual = iteration_handle.fetch_all()+1
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
Zraw = I + 1j * Q

np.savez_compressed(
    fpath, Navg=Navg, Nactual=Nactual, Zraw=Zraw,
    drivedelay_ns=drivedelay*4,
    config=config.meta, Vgate=Vgate)

lineI.set_ydata(I/config.short_readout_len*2**12)
lineQ.set_ydata(Q/config.short_readout_len*2**12)
ax.axvline(drivedelay*4, color='silver', linewidth=1, zorder=0)
ax.axvline(drivedelay*4+config.saturation_len, color='silver', linewidth=1, zorder=0)
ax2.plot(t, np.abs(Zraw))
ax3.plot(t, np.unwrap(np.angle(Zraw)))
ax2.set_ylabel('|Z|')
ax3.set_ylabel('arg Z')
readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"resonator {(config.resonatorLO+config.resonatorIF)/1e9:f}GHz"
    f"   qubit {(config.qubitLO)/1e9:.3f}GHz+{config.qubitIF/1e6:.2f}MHz"
    f"   Navg {Nactual}"
    f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
    f"\n{config.saturation_len}ns saturation pulse at {saturationpower:.1f}dBm{config.qubit_output_gain:+.1f}dB"
    f"\n{4*chunksize}ns chunks   {chunksperwindow} chunks per window   drive starts at {drivedelay*4}ns"
    f"\nVgate={Vgate:.6f}V")
fig.suptitle(title, fontsize=8)
fig.tight_layout()
fig.savefig(fpath+'.png', dpi=300)
#%%
plt.figure()
plt.plot(I, Q)