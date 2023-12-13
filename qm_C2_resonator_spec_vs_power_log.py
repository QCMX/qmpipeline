
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# Use this script to determine
# - electrical delay correction (config.PHASE_CORR)
# - resonator IF

import time
import numpy as np
import matplotlib.pyplot as plt
import importlib
import qm.qua as qua
from qualang_tools.loops import from_array
from helpers import data_path, mpl_pause, plt2dimg

import configuration as config
import qminit

qmm = qminit.connect()

#%%
# Config depends on experiment cabling
# OPX imported from config in extra file, Octave config here. :/
# Octave config is persistent even when opening new qm

importlib.reload(config)
importlib.reload(qminit)

withdrive = False

filename = '{datetime}_qm_resonator_vs_power'+('_withdrive' if withdrive else '')
fpath = data_path(filename, datesuffix='_qm')

Navg = 500

f_min = 203e6
f_max = 210e6
df = 0.05e6
freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs

# da = 0.002
# a_min = 0.002
# a_max = 1.0 # relative to readout amplitude
# amps = np.arange(a_min, a_max + da/2, da)  # + da/2 to add a_max to amplitudes
amps = np.logspace(np.log10(0.0316), np.log10(1), 101)
ampsV = amps*config.readout_amp

if withdrive:
    assert config.saturation_len > config.short_readout_len
    readoutwait = int(((config.saturation_len - config.short_readout_len) / 2) / 4) # cycles
    print("Readoutwait", readoutwait*4, "ns")

with qua.program() as resonator_spec:
    n = qua.declare(int)
    f = qua.declare(int)
    a = qua.declare(float)
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    n_st = qua.declare_stream()
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(*from_array(a, amps)):
            with qua.for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
                qua.update_frequency('resonator', f)  # update frequency of resonator element
                qua.wait(config.cooldown_clk, 'resonator')  # wait for resonator to decay
                qua.wait(rand.rand_int(50)+4, 'resonator')
                if withdrive:
                    qua.align()
                    qua.play('saturation', 'qubit')
                    qua.wait(readoutwait, 'resonator')
                qua.measure('readout'*qua.amp(a), 'resonator', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)
            qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(amps), len(freqs)).average().save('I')
        Q_st.buffer(len(amps), len(freqs)).average().save('Q')


# Execute program
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config)
if withdrive:
    qminit.octave_setup_qubit(qm, config)
job = qm.execute(resonator_spec)
tstart = time.time()

res_handles = job.result_handles  # get access to handles
I_handle = res_handles.get('I')
I_handle.wait_for_values(1)
Q_handle = res_handles.get('Q')
Q_handle.wait_for_values(1)
iteration_handle = res_handles.get('iteration')
iteration_handle.wait_for_values(1)

# Live plotting
fig, axs = plt.subplots()
artists = []
fig.suptitle('resonator spectroscopy analysis', fontsize=10)
axs.set_xlabel('IF')
axs.set_ylabel('|S|')
try:
    while res_handles.is_processing():    
        iteration = iteration_handle.fetch_all() + 1
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        Z = I + Q*1j
        for a in artists:
            a.remove()
        artists = []
        for i in range(len(amps)):
            artists.append(axs.plot(freqs, np.abs(Z[i]), color=plt.cm.rainbow(i/len(amps)))[0])
        axs.relim()
        axs.autoscale()
        axs.autoscale_view()
        fig.suptitle(f'{iteration} / {Navg}', fontsize=10)
        print(f"n={iteration}, remaining: {(Navg-iteration) * (time.time()-tstart)/iteration:.0f}s")
        #plt.pause(0.5)
        mpl_pause(0.5)
except KeyboardInterrupt:
    job.halt()
print("Execution time", time.time()-tstart, "s")

# Get and save data
Nactual = iteration_handle.fetch_all()+1
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
Zraw = I + 1j * Q

def amp2pow(amp):
    return 10*np.log10((config.readout_amp*amp)**2 * 10) + config.resonator_output_gain
def pow2amp(pow):
    return np.sqrt(10**((pow-config.resonator_output_gain)/10) / 10) / config.readout_amp

np.savez_compressed(
    fpath, Navg=Navg, Nactual=Nactual, Zraw=Zraw, freqs=freqs,
    amps=amps, ampsV=ampsV, power=amp2pow(amps),
    withdrive=withdrive, config=config.meta)

# Final plot
for a in artists:
    a.remove()
artists = []
for i in range(len(amps)):
    artists.append(axs.plot(freqs, np.abs(Z[i]), color=plt.cm.rainbow(i/len(amps)))[0])
axs.grid(), fig.tight_layout()


Zcorr = Zraw * np.exp(1j * freqs * config.PHASE_CORR) / config.readout_len * 2**12
absZ = np.abs(Zcorr)
magZ = 10*np.log10(np.abs(Zcorr/amps[:,None])**2 * 10)
argZ = np.unwrap(np.unwrap(np.angle(Zcorr))[::-1], axis=0)[::-1]
absZnoise = np.std(np.diff(absZ), axis=1) # noise
fitwhere = (np.max(absZ, axis=1) - np.min(absZ, axis=1)) > 7*absZnoise

readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    ("WITH QUBIT DRIVE\n" if withdrive else "")+
    f"LO={config.resonatorLO/1e9:.3f}GHz"
    f"   Navg {Nactual}"
    f"   electric delay {config.PHASE_CORR:.3e}rad/Hz"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain"
    +(f"\n{config.saturation_len}ns saturation pulse at {saturationpower:.1f}dBm{config.qubit_output_gain:+.1f}dB" if withdrive else ""))

# 2d maps
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, layout='constrained')
img = axs[0].pcolormesh(freqs, amp2pow(amps), magZ)
fig.colorbar(img, ax=axs[0], orientation='horizontal').set_label('|S| / dB')
img = axs[1].pcolormesh(freqs, amp2pow(amps), argZ)
fig.colorbar(img, ax=axs[1], orientation='horizontal').set_label('arg S / rad')
axs[0].set_xlabel('f1 / Hz')
axs[1].set_xlabel('f1 / Hz')
axs[0].set_ylabel('output power at octave / dBm')
secax = axs[1].secondary_yaxis('right', functions=(pow2amp, amp2pow))
secax.set_ylabel('relative pulse amplitude', fontsize=8)
fig.suptitle(title, fontsize=8)
fig.savefig(fpath+'_2dmap.png')

# Fit
def lorentzian(f, f0, width, a, tau0):
    tau = 0
    L = (width/2) / ((width/2) + 1j*(f - f0))
    return (a * np.exp(1j*(tau0 + tau*(f-f0))) * L).view(float)

from scipy.optimize import curve_fit
from uncertainties import ufloat

popt = None
popts, perrs = [], []
colors = np.array([plt.cm.rainbow(i / len(amps)) for i in range(len(amps))])
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), gridspec_kw={'width_ratios': [2,1]})
axs[0,0].sharex(axs[1,0])
axs[0,1].axis('equal')
for i in range(len(amps)):
    c = colors[i]
    axs[0,0].plot(freqs/1e6, 10*np.log10((np.abs(Zcorr[i])/amps[i])**2 * 10), color=c)
    axs[1,0].plot(freqs/1e6, np.unwrap(np.angle(Zcorr[i])), color=c)
    axs[0,1].plot(Zcorr[i].real, Zcorr[i].imag, color=c)

    if fitwhere[i]:
        print(f"Fit ({i}): amp {amps[i]}")
        if popt is None:
            p0 = [freqs[np.argmax(absZ[i])]/1e6, 0.45, np.mean(absZ[i]),0]# np.mean(np.unwrap(np.angle(Zcorr[i])))]
        else:
            p0 = popt
        popt, pcov = curve_fit(
            lorentzian, freqs/1e6, Zcorr[i].view(float), p0=p0)
        res = [ufloat(opt, err) for opt, err in zip(popt, np.sqrt(np.diag(pcov)))]
        for r, name in zip(res, ["f0", "width", "a", "tau", "tau0"]):
            print(f"  {name:6s} {r}")
        popts.append(popt)
        perrs.append(np.sqrt(np.diag(pcov)))

        model = lorentzian(freqs/1e6, *popt).view(complex)
        #ax1.set_title(f"fr={res[0]}MHz  width={res[1]}MHz  amp={10*np.log10(10*abs(res[2].nominal_value)**2):.0f}dBm", fontsize=8)
        axs[0,0].plot(freqs/1e6, 10*np.log10(10*np.abs(model/amps[i])**2), 'k-', linewidth=0.8, label="fit")
        axs[1,0].plot(freqs/1e6, np.unwrap(np.angle(model)), 'k-', linewidth=0.8)
        axs[0,1].plot(model.real, model.imag, 'k-', linewidth=0.8, zorder=-1)

popts, perrs = np.array(popts), np.array(perrs)
axs[1,1].scatter(amp2pow(amps[fitwhere]), popts[:,0], c=colors[fitwhere])
axs[1,1].errorbar(amp2pow(amps[fitwhere]), popts[:,0], yerr=perrs[:,0], fmt=' ', capsize=2, color='k', linewidth=1)
# def amp2pow(amp):
#     return 10*np.log10((config.readout_amp*amp)**2 * 10) + config.resonator_output_gain
# def pow2amp(pow):
#     return np.sqrt(10**((pow-config.resonator_output_gain)/10) / 10) / config.readout_amp
# secax = axs[1,1].secondary_xaxis('top', functions=(amp2pow, pow2amp))
# secax.set_xlabel('output power at octave / dBm', fontsize=8)

for ax in axs.flat: ax.grid()
axs[0,0].set_ylabel("|S / amp| / dB", fontsize=8)
axs[1,0].set_ylabel('arg S / rad', fontsize=8)
axs[1,0].set_xlabel('Resonator IF / MHz', fontsize=8)
axs[0,1].set_ylabel('Im S', fontsize=8), axs[0,1].set_xlabel('Re S', fontsize=8)
axs[1,1].set_ylabel('fr IF / MHz', fontsize=8), axs[1,1].set_xlabel('octave output power / dBm', fontsize=8)
fig.suptitle(title, fontsize=8)
fig.tight_layout()
fig.savefig(fpath+'.png')

# Fit results
goodamps = amp2pow(amps[fitwhere])
colors = np.array([plt.cm.rainbow(i / len(amps)) for i in range(len(amps))])[fitwhere]
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(8,5))
axs[0,0].scatter(goodamps, popts[:,0], c=colors)
axs[0,1].scatter(goodamps, popts[:,1]*1e3, c=colors)
axs[1,0].scatter(goodamps, popts[:,2]/amps[fitwhere], c=colors)
axs[1,1].scatter(goodamps, popts[:,3], c=colors)
axs[0,0].set_ylabel("fr / MHz")
axs[0,1].set_ylabel("width / kHz")
axs[1,0].set_ylabel("amp / amp [linear]")
axs[1,1].set_ylabel("tau / rad")
axs[1,0].set_xlabel("octave output power / dBm")
for ax in axs.flat: ax.grid()
fig.suptitle(title, fontsize=8)
fig.tight_layout()
fig.savefig(fpath+'_fit.png')

#%%

fig, axs = plt.subplots()
for i in range(len(amps)):
    c = plt.cm.rainbow(i / len(amps))
    axs.plot(Zcorr[i].real, Zcorr[i].imag, color=c)
axs.axis('equal')
axs.grid()

#%%


Zcorr = Zraw * np.exp(1j * freqs * config.PHASE_CORR) / config.readout_len * 2**12
absZ = np.abs(Zcorr)
absZnoise = np.std(np.diff(absZ), axis=1) # noise
fitwhere = (np.max(absZ, axis=1) - np.min(absZ, axis=1)) > 5*absZnoise

def cavity_pulse(f, f0, width, a, tau0, pulselen):
    # f in Hz, width in Hz, pulselen in seconds
    # Magic factor 4 for pulse length?
    x = width/2 + 1j*(f - f0)
    tau = 0
    L = width/2 / x * (np.exp(-x * pulselen*4) - 1)
    norm = (np.exp(-width/2 * pulselen*4) - 1)
    return (a / norm * np.exp(1j*(tau0 + tau*(f-f0))) * L).view(float)

from scipy.optimize import curve_fit
from uncertainties import ufloat

popt = None
popts, perrs = [], []
colors = np.array([plt.cm.rainbow(i / len(amps)) for i in range(len(amps))])
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), gridspec_kw={'width_ratios': [2,1]})
axs[0,0].sharex(axs[1,0])
axs[0,1].axis('equal')
for i in range(len(amps)):
    c = colors[i]
    axs[0,0].plot(freqs/1e6, 10*np.log10((np.abs(Zcorr[i])/amps[i])**2 * 10), color=c)
    axs[1,0].plot(freqs/1e6, np.unwrap(np.angle(Zcorr[i])), color=c)
    axs[0,1].plot(Zcorr[i].real, Zcorr[i].imag, color=c)

    if fitwhere[i]:
        # Fit freq in MHz, pulselen in us
        print(f"Fit ({i}): amp {amps[i]}")
        if popt is None:
            p0 = [freqs[np.argmax(absZ[i])]/1e6, 0.45, np.max(absZ[i]), np.mean(np.unwrap(np.angle(Zcorr[i]))), config.readout_len*1e-9*1e6]
            axs[0,0].plot(freqs/1e6, 10*np.log10(10*np.abs(cavity_pulse(freqs/1e6, *p0).view(complex)/amps[i])**2), 'k--', linewidth=0.8, label="fit")
            axs[1,0].plot(freqs/1e6, np.unwrap(np.angle(cavity_pulse(freqs/1e6, *p0).view(complex))), 'k--', linewidth=0.8)
        else:
            p0 = popt
        popt, pcov = curve_fit(
            cavity_pulse, freqs/1e6, Zcorr[i].view(float), p0=p0)
        res = [ufloat(opt, err) for opt, err in zip(popt, np.sqrt(np.diag(pcov)))]
        for r, name in zip(res, ["f0", "width", "a", "tau0", "pulselen"]):
            print(f"  {name:6s} {r}")
        popts.append(popt)
        perrs.append(np.sqrt(np.diag(pcov)))

        model = cavity_pulse(freqs/1e6, *popt).view(complex)
        #ax1.set_title(f"fr={res[0]}MHz  width={res[1]}MHz  amp={10*np.log10(10*abs(res[2].nominal_value)**2):.0f}dBm", fontsize=8)
        axs[0,0].plot(freqs/1e6, 10*np.log10(10*np.abs(model/amps[i])**2), 'k-', linewidth=0.8, label="fit")
        axs[1,0].plot(freqs/1e6, np.unwrap(np.angle(model)), 'k-', linewidth=0.8)
        axs[0,1].plot(model.real, model.imag, 'k-', linewidth=0.8, zorder=-1)

popts, perrs = np.array(popts), np.array(perrs)
axs[1,1].scatter(amps[fitwhere], popts[:,0], c=colors[fitwhere])
axs[1,1].errorbar(amps[fitwhere], popts[:,0], yerr=perrs[:,0], fmt=' ', capsize=2, color='k', linewidth=1)
def amp2pow(amp):
    return 10*np.log10((config.readout_amp*amp)**2 * 10) + config.resonator_output_gain
def pow2amp(pow):
    return np.sqrt(10**((pow-config.resonator_output_gain)/10) / 10) / config.readout_amp
secax = axs[1,1].secondary_xaxis('top', functions=(amp2pow, pow2amp))
secax.set_xlabel('output power at octave / dBm', fontsize=8)

for ax in axs.flat: ax.grid()
axs[0,0].set_ylabel("|S / amp| / dB", fontsize=8)
axs[1,0].set_ylabel('arg S / rad', fontsize=8)
axs[1,0].set_xlabel('Resonator IF / MHz', fontsize=8)
axs[0,1].set_ylabel('Im S', fontsize=8), axs[0,1].set_xlabel('Re S', fontsize=8)
axs[1,1].set_ylabel('fr IF / MHz', fontsize=8), axs[1,1].set_xlabel('relative pulse amplitude', fontsize=8)
readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
fig.suptitle(
    f"LO={config.resonatorLO/1e9:.3f}GHz"
    f"   Navg {Nactual}"
    f"   electric delay {config.PHASE_CORR:.3e}rad/Hz"
    f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain",
    fontsize=10)
fig.tight_layout()
fig.savefig(fpath+'_fitsinc.png')

