
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

import configuration_7GHz as config
import qminit

qmm = qminit.connect(use_calibration=False)

#%%

from instruments.basel import BaselDACChannel

#gate = BaselDACChannel(8) # 5 GHz
gate = BaselDACChannel(7) # 7 GHz

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 50e-6
GATERAMP_STEPTIME = 0.02

#%%

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_resonator_stat_vs_gate_7GHz'
fpath = data_path(filename, datesuffix='_qm')

Vgate = np.concatenate([np.linspace(-8.4, -8.0, 4001)])
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

Npoints = 500

f_min = 199e6
f_max = 203e6
df = 0.2e6
freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs
Nf = len(freqs)

with qua.program() as resonator_spec:
    n = qua.declare(int)
    f = qua.declare(int)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    n_st = qua.declare_stream()
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()

    with qua.for_(n, 0, n < Npoints, n + 1):
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
        I_st.with_timestamps().save_all('I')
        Q_st.save_all('Q')

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
progid = qm.compile(resonator_spec)
print("Program ID", progid)

print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)
print("Wait for gate to settle")
time.sleep(5)

dataS21 = np.full((Ngate, Npoints, Nf), np.nan+0j)
datatiming = np.full((Ngate, Npoints), np.nan)
tracetime = np.full(Ngate, np.nan)

fig, ax = plt.subplots()
img = plt2dimg(ax, Vgate, freqs, np.abs(dataS21[:,0,:]))
ax.set_xlabel("Vgate / V")
ax.set_ylabel("resonator IF")
#ax.yaxis.set_major_formatter(EngFormatter(sep='', unit='Hz'))
fig.colorbar(img, ax=ax).set_label("|S| / linear")
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"resonatorLO {(config.resonatorLO)/1e9:.5f}GHz"
    f"   Npoints {Npoints}"
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
        I = res_handles.get('I').fetch_all()['value'].reshape(Npoints, Nf)
        Q = res_handles.get('Q').fetch_all()['value'].reshape(Npoints, Nf)
        datatiming[i] = res_handles.get('I').fetch_all()['timestamp'].reshape(Npoints, Nf)[:,0]
        dataS21[i] = I + 1j * Q
        estimator.step(i)
        #plt2dimg_update(img, np.unwrap(np.unwrap(np.angle(dataS21), axis=1), axis=0))
        plt2dimg_update(img, np.abs(dataS21[:,0,:]))
finally:
    job.halt()
    estimator.end()
    np.savez_compressed(
        fpath, Npoints=Npoints, f=freqs, dataS21=dataS21, datatiming=datatiming,
        Vgate=Vgate,
        config=config.meta)
    print("Time per trace:", (np.nanmax(tracetime)-np.nanmin(tracetime))/np.count_nonzero(~np.isnan(tracetime)), 's')
    print("Time for gate set:", np.mean(tgate), "s")
    print("Time for QM execution:", np.mean(tqm), "s")

    plt2dimg_update(img, np.abs(dataS21[:,0,:]))
    fig.tight_layout()
    #fig.savefig(fpath+'.png')

#%%
# Fit data

from scipy.optimize import curve_fit
def lorentzian(f, f0, width, a, tau0):
    tau = 0
    L = (width/2) / ((width/2) + 1j*(f - f0))
    return (a * np.exp(1j*(tau0 + tau*(f-f0)/1e9)) * L).view(float)

popt = np.full((Ngate, Npoints, 4), np.nan)
p0 = None
for i in range(Ngate):
    print(i, "of", Ngate)
    for j in range(Npoints):
        Zcorr = dataS21[i,j] * np.exp(1j * freqs * config.PHASE_CORR) / config.readout_len * 2**12
        absZ = np.abs(Zcorr)
        if p0 is None:
            p0 = [freqs[np.argmax(absZ)]/1e6, 0.7, np.mean(absZ), np.mean(np.unwrap(np.angle(Zcorr)))]
        try:
            popt[i, j], pcov = curve_fit(lorentzian, freqs/1e6, Zcorr.view(float), p0=p0)
            if i == 2:
                p0 = popt[i,j]
        except RuntimeError as e:
            print(e)

np.savez_compressed(fpath+'_fit', popt=popt, Vgate=Vgate)

#%%

fr = popt[...,0]
binedges = np.linspace(199.6, np.nanmax(fr), 101)
bincenter = binedges[:-1] + np.diff(binedges)
frhist = np.array([np.histogram(popt[i], binedges)[0] for i in range(Ngate)])

fig, ax = plt.subplots()
img = plt2dimg(ax, Vgate, bincenter, frhist)
ax.set_xlabel("Vgate / V")
ax.set_ylabel(f"fr / MHz  ({len(bincenter)} bins)")
fig.colorbar(img, ax=ax).set_label('histogram')
fig.suptitle(title, fontsize=10)
fig.savefig(fpath+'_fr_hist.png')

#%%

#Shuttle
Vtarget = -9.71 # 6.6173

step = 50e-6
steptime = 0.02
print("Ramp time:", np.abs(Vtarget - gate.get_voltage()) / step * steptime / 60, "min")
gate.ramp_voltage(Vtarget, step, steptime)
