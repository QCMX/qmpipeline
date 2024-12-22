
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

import configuration as config
import qminit

qmm = qminit.connect()

#%%

from instruments.basel import BaselDACChannel

# gate = BaselDACChannel(8) # 5 GHz
#gate = BaselDACChannel(7) # 7 GHz

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 50e-6
GATERAMP_STEPTIME = 0.02

#%%

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_iqvsbaselgate'
fpath = data_path(filename, datesuffix='_qm')


Vgate = np.concatenate([np.linspace(-4.227, -4.233, 11)])
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
#assert Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

nbins = 1000
binedges = np.arange(nbins+1)/nbins * 2*np.pi - np.pi
bincenter = binedges[:-1] + np.diff(binedges)

Npoints = 20000

with qua.program() as iqpoints:
    n = qua.declare(int)
    n_st = qua.declare_stream()
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Npoints, n + 1):
        qua.reset_phase('resonator')
        qua.wait(rand.rand_int(50)+4, 'resonator')
        qua.wait(config.cooldown_clk, 'resonator')
        #play('saturation', 'qubit')
        # qua.align()
        qua.measure('readout', 'resonator', None,
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
progid = qm.compile(iqpoints)
print("Program ID", progid)

#if gate.get_voltage() != Vgate[0]: # oftenfalse due to uneven 
print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)
print("Wait for gate to settle")
time.sleep(5)

dataS = np.full((Ngate, Npoints), np.nan+0j)
dataStiming = np.full((Ngate, Npoints), np.nan)
datahist = np.full((Ngate, nbins), np.nan)
tracetime = np.full(Ngate, np.nan)

fig, ax = plt.subplots()
img = plt2dimg(ax, Vgate, bincenter, datahist)
ax.set_xlabel("Vgate / V")
ax.set_ylabel("arg S")
#ax.yaxis.set_major_formatter(EngFormatter(sep='', unit='Hz'))
fig.colorbar(img, ax=ax).set_label("Histogram")
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
title = (
    f"LO={config.resonatorLO/1e9:.5f}GHz   IF={config.resonatorIF/1e6:.5f}MHz   Npoints {Npoints}\n"
    f"{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain")
fig.suptitle(title, fontsize=8)
fig.tight_layout()
fig.show()

tgate = []
tqm = []
estimator = DurationEstimator(Ngate)
try:
    for i in range(Ngate):
        tstart = time.time()
        gate.ramp_voltage(Vgate[i], GATERAMP_STEP, GATERAMP_STEPTIME)
        tgate.append(time.time()-tstart)
        tracetime[i] = time.time()

        tstart = time.time()
        pendingjob = qm.queue.add_compiled(progid)
        job = pendingjob.wait_for_execution()
        res_handles = job.result_handles
        while res_handles.is_processing():
            mpl_pause(0.02)
        #res_handles.wait_for_all_values()
        tqm.append(time.time()-tstart)
        I_ = res_handles.get('I').fetch_all()
        I, Q = I_['value'], res_handles.get('Q').fetch_all()['value']
        dataStiming[i] = I_['timestamp'] # ns
        dataS[i] = I + 1j * Q
        datahist[i], _ = np.histogram(np.angle(dataS[i]), binedges)
        estimator.step(i)
        plt2dimg_update(img, datahist)
finally:
    job.halt()
    estimator.end()
    np.savez_compressed(
        fpath, Npoints=Npoints, Vgate=Vgate,
        dataS21=dataS, dataStiming_ns=dataStiming,
        datahistogram=datahist,
        tracetime=tracetime, config=config.meta)
    print("Time per trace:", (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime)), 's')
    print("Time for gate set:", np.mean(tgate), "s")
    print("Time for QM execution:", np.mean(tqm), "s")

    plt2dimg_update(img, datahist)
    fig.tight_layout()
    fig.savefig(fpath+'.png', dpi=300)

#%%
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
    f"LO={config.resonatorLO/1e9:.5f}GHz   IF={config.resonatorIF/1e6:.5f}MHz   Npoints per Vg {Npoints}\n"
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
    t = I_handle.fetch_all()['timestamp'] # ns
    I = I_['value']
    Q = Q_handle.fetch_all()['value']
    l = min(I.size, Q.size)
    Zraw = (I[:l] + Q[:l]*1j)
    Zcorr = Zraw / config.readout_len * 2**12
    
    np.savez_compressed(
        fpath, Npoints=Npoints, I=I, Q=Q, Zraw=Zraw, timing_ns=t,
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
    fig.tight_layout()
    fig.savefig(fpath, dpi=300)

#%%

fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(t[:Zcorr.size]/1e3, Zcorr.real)
axs[1].plot(t[:Zcorr.size]/1e3, Zcorr.imag)
axs[1].set_xlabel("Time / us")
