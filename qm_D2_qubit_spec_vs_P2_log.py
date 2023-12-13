
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
from qualang_tools.loops import from_array

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

filename = '{datetime}_qm_qubit_vs_power2'
fpath = data_path(filename, datesuffix='_qm')

Navg = 1000

f_min = 50e6
f_max = 450e6
df = 2e6
freqs = int_forloop_values(f_min, f_max, df)

amps = amps = np.logspace(np.log10(0.0316), np.log10(1), 51)
ampsV = amps*config.saturation_amp

try:
    Vgate = gate.get_voltage()
except:
    Vgate = np.nan

# Align readout pulse in middle of saturation pulse
assert config.saturation_len > config.short_readout_len
readoutwait = int(((config.saturation_len - config.short_readout_len) / 2) / 4) # cycles
print("Readoutwait", readoutwait*4, "ns")

with qua.program() as qubit_vs_power:
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
            with qua.for_(f, f_min, f <= f_max, f + df):
                qua.update_frequency('qubit', f)
                qua.wait(config.cooldown_clk, 'resonator')
                qua.wait(rand.rand_int(50)+4, 'resonator')
                qua.align()
                qua.play('saturation'*qua.amp(a), 'qubit')
                qua.wait(readoutwait, 'resonator')
                qua.measure('short_readout', 'resonator', None,
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
axs.set_xlabel('qubit IF')
axs.set_ylabel('arg S')
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
            artists.append(axs.plot(freqs, np.angle(Z[i]), color=plt.cm.rainbow(i/len(amps)))[0])
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
    return 10*np.log10((config.saturation_amp*amp)**2 * 10) + config.qubit_output_gain
def pow2amp(pow):
    return np.sqrt(10**((pow-config.qubit_output_gain)/10) / 10) / config.saturation_amp

np.savez_compressed(
    fpath, Navg=Navg, freqs=freqs, Zraw=Zraw,
    amps=amps, ampsV=ampsV, power=amp2pow(amps),
    Nactual=Nactual, config=config.meta)

# Final live plot
for a in artists:
    a.remove()
artists = []
for i in range(len(amps)):
    artists.append(axs.plot(freqs, np.angle(Z[i]), color=plt.cm.rainbow(i/len(amps)))[0])
axs.grid(), fig.tight_layout()

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
axs[1,1].scatter(amp2pow(amps), argZ[:,fqidx], c=colors)
for ax in axs.flat: ax.grid()
axs[0,0].set_ylabel("|S| / dBm", fontsize=8)
axs[1,0].set_ylabel('arg S / rad', fontsize=8)
axs[1,0].set_xlabel('Resonator IF / MHz', fontsize=8)
axs[0,1].set_ylabel('Im S', fontsize=8), axs[0,1].set_xlabel('Re S', fontsize=8)
axs[1,1].set_ylabel(f'arg S at {config.qubitIF/1e6:.3f}MHz', fontsize=8)
axs[1,1].set_xlabel('drive output power / dBm', fontsize=8)

secax = axs[1,1].secondary_xaxis('top', functions=(pow2amp, amp2pow))
secax.set_xlabel('relative amplitude', fontsize=8)
#axs[1,1].set_ylabel('fr IF / MHz', fontsize=8), axs[1,1].set_xlabel('relative pulse amplitude', fontsize=8)
readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"readout {config.resonatorLO/1e9:.3f}GHz+{config.resonatorIF/1e6:.2f}MHz"
    f"   qubit LO {config.qubitLO/1e9:.3f}GHz"
    f"   Navg {Nactual}"
    f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
    f",   {config.resonator_input_gain:+.1f}dB input gain"
    f"\n{config.saturation_len}ns saturation pulse at {saturationpower:.1f}dBm{config.qubit_output_gain:+.1f}dB"
    f"\nVgate={Vgate:.7f}V")
fig.suptitle(title, fontsize=8)
fig.tight_layout()
fig.savefig(fpath+'.png')


fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, layout='constrained')
im = axs[0].pcolormesh(freqs/1e6, amp2pow(amps), magZ)
fig.colorbar(im, orientation='horizontal').set_label('|S| / dBm')
im = axs[1].pcolormesh(freqs/1e6, amp2pow(amps), argZ)
fig.colorbar(im, orientation='horizontal').set_label('arg Z')
axs[0].set_ylabel('P2 octave output / dBm')
axs[0].set_xlabel('IF / MHz')
axs[1].set_xlabel('IF / MHz')
fig.suptitle(title, fontsize=8)
fig.savefig(fpath+'_2d.png')
