
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

filename = '{datetime}_qm_qubit_spec_delay'
fpath = data_path(filename, datesuffix='_qm')

Navg = 200000

# needs to be in range 4 to 2**31-1 cycles
# 20ns readout = 5cycles
delay_min = 4//4 # cycles
delay_max = 1500//4 # cycles
delay_step = 4//4 # cycles
delays = int_forloop_values(delay_min, delay_max, delay_step)
delays_ns = delays * 4

saturationdelay = 300//4 #cycles

try:
    Vgate = gate.get_voltage()
except:
    Vgate = np.nan

with qua.program() as qubit_delay:
    n = qua.declare(int)  # variable for average loop
    n_st = qua.declare_stream()  # stream for 'n'
    delay = qua.declare(int)
    f = qua.declare(int)
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    I_st = qua.declare_stream()  # stream for I
    Q_st = qua.declare_stream()  # stream for Q
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(delay, delay_min, delay <= delay_max, delay + delay_step):
            # qua.reset_phase('resonator')
            # qua.reset_phase('qubit')
            qua.wait(rand.rand_int(50)+4, 'resonator')
            qua.align()
            qua.wait(saturationdelay, 'qubit')
            qua.play('saturation', 'qubit')
            qua.wait(delay, 'resonator')
            qua.measure('short_readout', 'resonator', None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
            qua.wait(config.cooldown_clk, 'resonator')
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(delays)).average().save('I')
        Q_st.buffer(len(delays)).average().save('Q')


# #%%
# from qm import LoopbackInterface, SimulationConfig
# simulate_config = SimulationConfig(
#     duration=20000, # cycles
#     simulation_interface=LoopbackInterface(([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])))
# job = qmm.simulate(config.qmconfig, qubit_delay, simulate_config)  # do simulation with qmm

# plt.figure()
# job.get_simulated_samples().con1.plot()  # visualize played pulses
# #plt.savefig(fpath+"_pulses.png")
# #%%

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config, short_readout_gain=True)
qminit.octave_setup_qubit(qm, config)

# Execute program
job = qm.execute(qubit_delay)
tstart = time.time()

res_handles = job.result_handles
I_handle = res_handles.get('I')
I_handle.wait_for_values(1)
Q_handle = res_handles.get('Q')
Q_handle.wait_for_values(1)
iteration_handle = res_handles.get('iteration')
iteration_handle.wait_for_values(1)

# Live plotting
data = np.full(len(delays), np.nan)
fig, ax = plt.subplots()
line, = ax.plot(delays_ns, np.angle(data))
ax.set_xlabel('delay / ns')
ax.set_ylabel('arg S')
ax.set_title('qubit analysis with delay')

try:
    while res_handles.is_processing():
        iteration = iteration_handle.fetch_all() + 1
        I = I_handle.fetch_all().T
        Q = Q_handle.fetch_all().T
        Z = I + Q*1j
        ax.set_title(f'{iteration}/{Navg}')
        line.set_ydata(np.unwrap(np.angle(Z)))
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
I = I_handle.fetch_all().T
Q = Q_handle.fetch_all().T
Zraw = I + 1j * Q

np.savez_compressed(
    fpath, Navg=Navg, Zraw=Zraw, Nactual=Nactual,
    delays_clk=delays, delays_ns=delays_ns, saturationdelay=saturationdelay,
    config=config.meta, Vgate=Vgate)

# Final plot
# arg = np.unwrap(np.unwrap(np.angle(Zraw), axis=1), axis=0)
line.set_ydata(np.unwrap(np.angle(Z)))
readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
title = (
    f"resonator {(config.resonatorLO+config.resonatorIF)/1e9:f}GHz"
    f"   qubit {(config.qubitLO)/1e9:.3f}GHz+{config.qubitIF/1e6:.2f}MHz"
    f"   Navg {Nactual}"
    f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
    f"\n{config.saturation_len}ns saturation pulse at {saturationpower:.1f}dBm{config.qubit_output_gain:+.1f}dB"
    f"\nVgate={Vgate:.6f}V")
fig.suptitle(title, fontsize=8)
fig.tight_layout()
plt.close(fig)

Zcorr = Zraw / config.short_readout_len * 2**12

fig, axs = plt.subplots(nrows=3, sharex=True, layout='constrained')
axs[0].fill_between(
    [delays_ns[0], delays_ns[-1]],
    [saturationdelay*4, saturationdelay*4],
    [saturationdelay*4+config.saturation_len, saturationdelay*4+config.saturation_len],
    color='C1', alpha=0.3, label='saturation')
axs[0].fill_between(delays_ns, delays_ns, delays_ns+config.short_readout_len, color='C0', alpha=0.3, label='readout')
axs[0].plot(delays_ns, delays_ns, '.-', color='C0', alpha=0.3, label='readout')

axs[1].plot(delays_ns, np.abs(Zcorr))
axs[2].plot(delays_ns, np.unwrap(np.angle(Zcorr)))
axs[0].legend(loc='lower right', fontsize=6)
axs[0].set_ylabel('time / ns')
axs[1].set_ylabel('|Z| (linear)')
axs[2].set_ylabel('arg Z')
axs[2].set_xlabel('delay / ns')

for ax in axs: ax.grid()
fig.suptitle(title, fontsize=8)

fig.savefig(fpath+'.png')

#%%
# Model: (lead), step down with risetime, (sustain), step up with falltime

from scipy.optimize import curve_fit
from uncertainties import ufloat

def model(delay, lead, sustain, risetime, falltime, base, amp):
    return base - amp * (
        np.maximum(0, np.minimum(1, (delay-lead)/risetime))
        - np.maximum(0, np.minimum(1, (delay-(lead+risetime+sustain))/falltime))
        )

arg = np.unwrap(np.unwrap(np.angle(Zcorr)), axis=0)

popts, perrs = [], []

base = np.mean(arg[-10:])
shortest = min(config.short_readout_len, config.saturation_len)
p0 = [
    saturationdelay*4-config.short_readout_len,
    max(1, abs(config.short_readout_len-config.saturation_len)),
    shortest, shortest, base, base - np.min(arg)]
popt, pcov = curve_fit(
    model, delays_ns, arg, p0=p0, bounds=(
        [-np.inf, 0, 0, 0, -np.inf, 0], np.inf
        ))
res = [ufloat(opt, err) for opt, err in zip(popt, np.sqrt(np.diag(pcov)))]
for r, name in zip(res, ["lead", "sustain", "risetime", "falltime", "base", "amp"]):
    print(f"  {name:8s} {r}")
perr = np.sqrt(np.diag(pcov))

fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(delays_ns, model(delays_ns, *p0), 'k--', linewidth=1, label='expected')

axs[0].plot(delays_ns, arg)
axs[0].plot(delays_ns, model(delays_ns, *popt), 'k-', linewidth=0.8)
axs[1].plot(delays_ns, arg-model(delays_ns, *popt))

#axs[0].plot(delays_ns, np.mean(arg, axis=1), 'C3.-', linewidth=2, label='avg')

axs[1].set_xlabel('delay / ns')
axs[0].legend(loc='lower right')
axs[0].set_ylabel('arg S')
axs[1].set_ylabel('residuals')
fig.suptitle(title+f"\nrisetime {str(res[2])} ns   falltime {str(res[3])}", fontsize=8)
fig.tight_layout()
plt.savefig(fpath+'_fit.png')
