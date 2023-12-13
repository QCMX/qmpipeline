
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# - resonator IF, readout power

import time
import numpy as np
import matplotlib.pyplot as plt
import importlib
import qm.qua as qua

from helpers import data_path, mpl_pause, plt2dimg
from qm_helpers import int_forloop_values

import configuration as config
import qminit

qmm = qminit.connect()

#%%
# Config depends on experiment cabling
# OPX imported from config in extra file, Octave config here. :/
# Octave config is persistent even when opening new qm

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_qubit_vs_P1_P2'
fpath = data_path(filename, datesuffix='_qm')

Navg = 1000 

da1 = 0.1 
a1_min = 0.02 
a1_max = 1.0 
amps1 = np.arange(a1_min, a1_max + da1/2, da1)
amps1V = amps1*config.short_readout_amp

da2 = 0.1 
a2_min = 0.02 
a2_max = 1.0 
amps2 = np.arange(a2_min, a2_max + da2/2, da2) # np.array([0.1,1.0]) #
amps2V = amps2*config.saturation_amp

try:
    Vgate = gate.get_voltage()
except:
    Vgate = None

# Align readout pulse in middle of saturation pulse
assert config.saturation_len > config.short_readout_len
readoutwait = int(((config.saturation_len - config.short_readout_len) / 2) / 4) # cycles
print("Readoutwait", readoutwait*4, "ns")

with qua.program() as qubit_vs_power:
    n = qua.declare(int)
    a1 = qua.declare(float)
    a2 = qua.declare(float)
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    n_st = qua.declare_stream()
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(a1, a1_min, a1 < a1_max + da1/2, a1 + da1):
            with qua.for_(a2, a2_min, a2 < a2_max + da2/2, a2 + da2):
                qua.wait(config.cooldown_clk, 'resonator')
                qua.wait(rand.rand_int(50)+4, 'resonator')
                qua.align()
                qua.play('saturation'*qua.amp(a2), 'qubit')
                qua.wait(readoutwait, 'resonator')
                qua.measure('short_readout'*qua.amp(a1), 'resonator', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)
            qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(amps1), len(amps2)).average().save('I')
        Q_st.buffer(len(amps1), len(amps2)).average().save('Q')


# Execute program
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config, short_readout_gain=True)
qminit.octave_setup_qubit(qm, config)
job = qm.execute(qubit_vs_power)
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
fig.suptitle('qubit spectroscopy analysis', fontsize=10)
axs.set_xlabel('qubit relative amp')
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
        for i in range(len(amps1)):
            artists.append(axs.plot(amps2, np.abs(Z[i]), color=plt.cm.rainbow(i/len(amps1)))[0])
        axs.relim()
        axs.autoscale()
        axs.autoscale_view()
        fig.suptitle(f'{iteration} / {Navg}', fontsize=10)
        print(f"n={iteration}, remaining: {(Navg-iteration) * (time.time()-tstart)/iteration:.0f}s")
        mpl_pause(0.5)
except KeyboardInterrupt:
    job.halt()
print("Execution time", time.time()-tstart, "s")

# Get and save data
Nactual = iteration_handle.fetch_all()+1
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
Zraw = I + 1j * Q

np.savez_compressed(
    fpath, Navg=Navg, Nactual=Nactual, amps1=amps1, amps2=amps2, Zraw=Zraw,
    config=config.meta)

# Final live plot
for a in artists:
    a.remove()
artists = []
for i in range(len(amps1)):
    artists.append(axs.plot(amps2, np.abs(Z[i]), color=plt.cm.rainbow(i/len(amps1)))[0])
axs.grid(), fig.tight_layout()

#%%

readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"LO={config.resonatorLO/1e9:.3f}GHz"
    f"   Navg {Nactual}"
    f"   electric delay {config.PHASE_CORR:.3e}rad/Hz"
    f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain"
    f"\n{config.saturation_len}ns saturation pulse at {saturationpower:.1f}dBm{config.qubit_output_gain:+.1f}dB"
    f"\nVgate={Vgate}V")

Zcorr = Zraw / amps1[:,None] / config.readout_len * 2**12
absZ = np.abs(Zcorr)
argZ = np.unwrap(np.angle(Zcorr))
magZ = 10*np.log10((np.abs(Zcorr))**2 * 10)

fig, axs = plt.subplots(nrows=2, ncols=2)
axs[1,0].sharex(axs[0,0])
axs[1,1].sharex(axs[0,1])
axs[0,1].sharey(axs[0,0])
axs[1,1].sharey(axs[1,0])
for i in range(len(amps1)):
    axs[0,0].plot(amps2, magZ[i], color=plt.cm.rainbow(i/len(amps1)), label=f"{amps1[i]}" if i==0 or i==len(amps2)-1 else None)
    axs[1,0].plot(amps2, argZ[i], color=plt.cm.rainbow(i/len(amps1)))
for i in range(len(amps2)):
    axs[0,1].plot(amps1, magZ[:,i], color=plt.cm.rainbow(i/len(amps2)), label=f"{amps2[i]}" if i==0 or i==len(amps2)-1 else None)
    axs[1,1].plot(amps1, argZ[:,i], color=plt.cm.rainbow(i/len(amps2)))
axs[0,0].set_ylabel('|S / readout amp| / dB')
axs[1,0].set_ylabel('arg S / rad')
axs[1,0].set_xlabel('relative drive amplitude')
axs[1,1].set_xlabel('relative readout amplitude')
axs[0,0].legend()
axs[0,1].legend()
fig.suptitle(title, fontsize=8)
fig.tight_layout()
fig.savefig(fpath+'_lines.png')

#%%

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, layout='constrained')
img = plt2dimg(axs[0], amps2, amps1, magZ.T)
fig.colorbar(img, ax=axs[0], orientation='horizontal').set_label('|S / readout amp| / dB')
img = plt2dimg(axs[1], amps2, amps1, argZ.T)
fig.colorbar(img, ax=axs[1], orientation='horizontal').set_label('arg S / rad')
axs[0].set_xlabel('relative drive amplitude')
axs[1].set_xlabel('relative drive amplitude')
axs[0].set_ylabel('relative readout amplitude')

def amp2pow(amp):
    return 10*np.log10((config.saturation_amp*amp)**2 * 10) + config.qubit_output_gain
def pow2amp(pow):
    return np.sqrt(10**((pow-config.qubit_output_gain)/10) / 10) / config.saturation_amp
secax = axs[1].secondary_xaxis('top', functions=(amp2pow, pow2amp))
secax = axs[0].secondary_xaxis('top', functions=(amp2pow, pow2amp))
secax.set_xlabel('drive output power at octave / dBm', fontsize=8)

def amp2pow(amp):
    return 10*np.log10((config.short_readout_amp*amp)**2 * 10) + config.resonator_output_gain
def pow2amp(pow):
    return np.sqrt(10**((pow-config.resonator_output_gain)/10) / 10) / config.short_readout_amp
secax = axs[1].secondary_yaxis('right', functions=(amp2pow, pow2amp))
secax.set_ylabel('readout power at octave / dBm', fontsize=8)
fig.suptitle(title, fontsize=8)
fig.savefig(fpath+'_2dmap.png')

#%%
# Nice plot

Zcorr = Zraw / config.readout_len * 2**12
absZ = np.abs(Zcorr)
argZ = np.unwrap(np.angle(Zcorr))
magZ = 10*np.log10((np.abs(Zcorr))**2 * 10)

colors = np.array([plt.cm.rainbow(i / len(amps)) for i in range(len(amps))])
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), gridspec_kw={'width_ratios': [2,1]})
axs[0,0].sharex(axs[1,0])
axs[0,1].axis('equal')
for i in range(len(amps)):
    c = colors[i]
    axs[0,0].plot(freqs/1e6, magZ[i], color=c)
    axs[1,0].plot(freqs/1e6, argZ[i], color=c)
    axs[0,1].plot(Zcorr[i].real, Zcorr[i].imag, color=c, linewidth=0.8)
fqidx = np.argmin(np.abs(freqs - config.qubitIF))
axs[1,1].scatter(amps, argZ[:,fqidx], c=colors)
for ax in axs.flat: ax.grid()
axs[0,0].set_ylabel("|S| / dBm", fontsize=8)
axs[1,0].set_ylabel('arg S / rad', fontsize=8)
axs[1,0].set_xlabel('Resonator IF / MHz', fontsize=8)
axs[0,1].set_ylabel('Im S', fontsize=8), axs[0,1].set_xlabel('Re S', fontsize=8)
axs[1,1].set_ylabel(f'|S| at {config.qubitIF/1e6:.3f}MHz', fontsize=8)
axs[1,1].set_xlabel('relative pulse amplitude', fontsize=8)
def amp2pow(amp):
    return 10*np.log10((config.saturation_amp*amp)**2 * 10) + config.qubit_output_gain
def pow2amp(pow):
    return np.sqrt(10**((pow-config.qubit_output_gain)/10) / 10) / config.saturation_amp
secax = axs[1,1].secondary_xaxis('top', functions=(amp2pow, pow2amp))
secax.set_xlabel('drive output power at octave / dBm', fontsize=8)
#axs[1,1].set_ylabel('fr IF / MHz', fontsize=8), axs[1,1].set_xlabel('relative pulse amplitude', fontsize=8)

fig.suptitle(title, fontsize=8)
fig.tight_layout()
fig.savefig(fpath+'.png')


