
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# - electrical delay correction (config.PHASE_CORR)
# - resonator IF

import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import qm.qua as qua

from helpers import data_path, mpl_pause, DurationEstimator, plt2dimg, plt2dimg_update

import configuration_novna as config
import qminit

qmm = qminit.connect()

#%%


importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_iqvstime'
fpath = data_path(filename, datesuffix='_qm')

Npoints = 5000000

with qua.program() as resonator_spec:
    n = qua.declare(int)
    n_st = qua.declare_stream()
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()

    with qua.for_(n, 0, n < Npoints, n + 1):
        qua.reset_phase('resonator')
        qua.wait(config.cooldown_clk, 'resonator')
        qua.play('pi', 'qubit')
        qua.align()
        qua.measure('short_readout', 'resonator', None,
                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
        qua.save(I, I_st)
        qua.save(Q, Q_st)
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.with_timestamps().save_all('I')
        Q_st.save_all('Q')

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
job = qm.execute(resonator_spec)

res_handles = job.result_handles
I_handle = res_handles.get('I')
Q_handle = res_handles.get('Q')
iteration_handle = res_handles.get('iteration')
I_handle.wait_for_values(1), Q_handle.wait_for_values(1)
iteration_handle.wait_for_values(1)

# Live plotting
fig, ax = plt.subplots()
ax.axis('equal')
dots, = ax.plot([0], [0], '.', ms=1, alpha=0.1)
ax.set_xlabel('I = Re S')
ax.set_ylabel('Q = Im S')
readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
title = (
    f"LO={config.resonatorLO/1e9:.5f}GHz   IF={config.resonatorIF/1e6:.5f}MHz   Npoints {Npoints}\n"
    f"{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain")
fig.suptitle(title, fontsize=10)
fig.show()
try:
    while res_handles.is_processing():
        iteration = iteration_handle.fetch_all() + 1
        I, Q = I_handle.fetch_all()['value'], Q_handle.fetch_all()['value']
        l = min(I.size, Q.size)
        Z = (I[:l] + Q[:l]*1j) / config.readout_len * 2**12
        dots.set_data(Z.real, Z.imag)
        ax.relim(), ax.autoscale(), ax.autoscale_view()
        print(iteration)
        mpl_pause(0.5)
except KeyboardInterrupt as e:
    job.halt()
    raise e
finally:
    Nactual = iteration_handle.fetch_all()+1
    I_ = I_handle.fetch_all()
    t = I_['timestamp'] # ns
    I = I_['value']
    Q = Q_handle.fetch_all()['value']
    l = min(I.size, Q.size)
    Zraw = (I[:l] + Q[:l]*1j)
    Zcorr = Zraw / config.readout_len * 2**12
    
    np.savez_compressed(
        fpath, Npoints=Npoints, I=I, Q=Q, Zraw=Zraw, timing_ns=t, Nactual=Nactual,
        config=config.meta)

    # Final plot
    dots.set_data(Zcorr.real, Zcorr.imag)
    ax.relim(), ax.autoscale(), ax.autoscale_view()
    ax.grid(), fig.tight_layout()
    fig.savefig(fpath, dpi=300)

    from matplotlib.colors import PowerNorm
    fig, ax = plt.subplots()
    h = ax.hist2d(Zcorr.real, Zcorr.imag, bins=1000, norm=PowerNorm(0.5))
    fig.colorbar(h[3], ax=ax).set_label("Histogram")
    ax.axis('equal')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(fpath, dpi=300)

#%%

fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(t[:Zcorr.size]/1e3, Zcorr.real)
axs[1].plot(t[:Zcorr.size]/1e3, Zcorr.imag)
axs[1].set_xlabel("Time / us")

#%%

plt.figure()
plt.hist(np.angle(Zcorr), bins=1000)