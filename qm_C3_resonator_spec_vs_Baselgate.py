
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

gate = BaselDACChannel(3) # 5 GHz
#gate = BaselDACChannel(7) # 7 GHz

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 50e-6
GATERAMP_STEPTIME = 0.02

#%%

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_resonator_vs_gate'
fpath = data_path(filename, datesuffix='_qm')

Vgate = np.concatenate([np.linspace(-4.35, -4.25, 501)])
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

Navg = 200

f_min = 205e6
f_max = 209e6
df = 0.05e6
freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs
Nf = len(freqs)

dataS21 = np.full((Ngate, Nf), np.nan+0j)
tracetime = np.full(Ngate, np.nan)

with qua.program() as resonator_spec:
    n = qua.declare(int)  # variable for average loop
    f = qua.declare(int)  # variable to sweep freqs
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    n_st = qua.declare_stream()  # stream for 'n'
    I_st = qua.declare_stream()  # stream for I
    Q_st = qua.declare_stream()  # stream for Q

    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(f, f_min, f <= f_max, f + df):  # integer loop
            qua.update_frequency('resonator', f)
            qua.wait(config.cooldown_clk, 'resonator')
            qua.measure('readout', 'resonator', None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(freqs)).average().save('I')
        Q_st.buffer(len(freqs)).average().save('Q')

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
progid = qm.compile(resonator_spec)
print("Program ID", progid)

print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)
print("Wait for gate to settle")
time.sleep(5)

fig, ax = plt.subplots()
img = plt2dimg(ax, Vgate, freqs, np.abs(dataS21))
ax.set_xlabel("Vgate / V")
ax.set_ylabel("resonator IF")
#ax.yaxis.set_major_formatter(EngFormatter(sep='', unit='Hz'))
fig.colorbar(img, ax=ax).set_label("|S| / linear")
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"resonatorLO {(config.resonatorLO)/1e9:.5f}GHz"
    f"   Navg {Navg}"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain")
fig.suptitle(title, fontsize=8)
fig.tight_layout()

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
        I = res_handles.get('I').fetch_all()
        Q = res_handles.get('Q').fetch_all()
        dataS21[i] = I + 1j * Q
        estimator.step(i)
        #plt2dimg_update(img, np.unwrap(np.unwrap(np.angle(dataS21), axis=1), axis=0))
        plt2dimg_update(img, np.abs(dataS21))
finally:
    job.halt()
    estimator.end()
    np.savez_compressed(
        fpath, Navg=Navg, f=freqs, dataS21=dataS21, Vgate=Vgate,
        config=config.meta)
    print("Time per trace:", (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime)), 's')
    print("Time for gate set:", np.mean(tgate), "s")
    print("Time for QM execution:", np.mean(tqm), "s")

    
    plt2dimg_update(img, np.abs(dataS21))
    fig.tight_layout()
    # fig.savefig(fpath+'.png', dpi=300)

    arg = np.unwrap(np.unwrap(np.angle(dataS21), axis=1), axis=0)

    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, layout='constrained')
    img = plt2dimg(axs[0], Vgate, freqs/1e6, 10*np.log10(np.abs(dataS21)**2 * 10))
    fig.colorbar(img, ax=axs[0]).set_label("|S| / dBm")
    img = plt2dimg(axs[1], Vgate, freqs/1e6, arg)
    fig.colorbar(img, ax=axs[1]).set_label("arg S")
    axs[0].set_ylabel('resonator IF / MHz', fontsize=6)
    axs[1].set_ylabel('resonator IF / MHz', fontsize=6)
    axs[1].set_xlabel("Vgate / V")
    fig.suptitle(title, fontsize=8)
    fig.savefig(fpath+'.png', dpi=300)

#%%

#Shuttle
Vtarget = -4.41 # 6.6173

step = 5e-6
steptime = 0.02
print("Ramp time:", np.abs(Vtarget - gate.get_voltage()) / step * steptime / 60, "min")
gate.ramp_voltage(Vtarget, step, steptime)
