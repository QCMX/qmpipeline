# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# - electrical delay correction (config.PHASE_CORR)
# - resonator IF, readout power
# - qubit IF, (drive power)

import time
import numpy as np
import matplotlib.pyplot as plt
import importlib

import qm.qua as qua

from helpers import data_path, mpl_pause
from qm_helpers import int_forloop_values

import configuration as config
import qminit

qmm = qminit.connect()

#%%
importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_power_rabi'
fpath = data_path(filename, datesuffix='_qm')


Navg = int(20e6)

a_min = 0.0
a_max = 1.0 # relative to gaussian amplitude
da = 0.005
amps = np.arange(a_min, a_max + da/2, da)  # + da/2 to add a_max to amplitudes
ampsV = amps*config.gauss_amp # in config.gauss_amp - config.const_amp
pulse = 'const' # 'gaussian'

with qua.program() as power_rabi:
    n = qua.declare(int)
    n_st = qua.declare_stream()
    a = qua.declare(qua.fixed)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(a, a_min, a < a_max + da/2, a + da):
            qua.play(pulse*qua.amp(a), 'qubit')
            qua.align()
            qua.measure('short_readout', 'resonator', None,
                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
            qua.wait(config.cooldown_clk, 'qubit')
            qua.wait(rand.rand_int(50)+4, 'resonator') # randomize demod error
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(amps)).average().save('I')
        Q_st.buffer(len(amps)).average().save('Q')

# #%%

# from qm import LoopbackInterface, SimulationConfig
# simulate_config = SimulationConfig(
#     duration=20000, # cycles
#     simulation_interface=LoopbackInterface(([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])))
# job = qmm.simulate(config.qmconfig, power_rabi, simulate_config)  # do simulation with qmm
# plt.figure()
# job.get_simulated_samples().con1.plot()  # visualize played pulses

# #%%

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config, short_readout_gain=True)
qminit.octave_setup_qubit(qm, config)


job = qm.execute(power_rabi)  # execute QUA program
tstart = time.time()

res_handles = job.result_handles  # get access to handles
I_handle = res_handles.get('I')
I_handle.wait_for_values(1)
Q_handle = res_handles.get('Q')
Q_handle.wait_for_values(1)
iteration_handle = res_handles.get('iteration')
iteration_handle.wait_for_values(1)

fig, ax = plt.subplots()
ax.set_xlabel('IF [Hz]')
ax.set_ylabel('arg (demod signal)')
ax.set_title('power rabi')
line, = ax.plot(amps, np.full(amps.size, np.nan))
fig.tight_layout()
try:
    while res_handles.is_processing():
        iteration = iteration_handle.fetch_all() + 1
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        Z = I + Q*1j
        line.set_ydata(np.angle(Z))
        ax.set_title(f'power rabi {iteration}/{Navg}')
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

res_handles.wait_for_all_values()
print(f"execution time: {time.time()-tstart:.1f}s")
print(job.execution_report())

Nactual = iteration_handle.fetch_all()+1
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
Zraw = I + 1j * Q

np.savez_compressed(
    fpath, Zraw=Zraw,
    Navg=Navg, Nactual=Nactual, amps=amps, ampsV=ampsV,
    config=config.meta)

plt.close(fig)

Zcorr = Zraw / config.short_readout_len * 2**12

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(amps, 10*np.log10((np.abs(Zcorr))**2 * 10))
ax2.plot(amps, np.unwrap(np.angle(Zcorr)))
ax1.grid(), ax2.grid()
ax1.set_ylabel('|S| / dB')
ax2.set_ylabel('arg S / rad')
ax2.set_xlabel('relative pulse amplitude')
readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
drivepower = 10*np.log10(config.gauss_amp**2 * 10) # V to dBm
fig.suptitle(
    f"resonator {(config.resonatorLO+config.resonatorIF)/1e9:f}GHz"
    f"   qubit {(config.qubitLO+config.qubitIF)/1e9:f}GHz"
    f"   Navg {Navg}"
    f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
    f"\n{config.gauss_len}ns {pulse} pulse at {drivepower:.1f}dBm{config.qubit_output_gain:+.1f}dB"
    f"\nVgate={Vgate:.6f}V",
    fontsize=8)
fig.tight_layout()
fig.savefig(fpath+'.png')

#%%

def harmonic(t, fV, A, B):
    return B + A * np.cos(2*np.pi * t * fV)

from scipy.optimize import curve_fit

p0 = [3, 1e-5, 0]
#plt.plot(ampsV, harmonic(ampsV, *p0), ':k')

popt, pcov = curve_fit(harmonic, ampsV, I, p0=p0)
print(popt)
plt.plot(ampsV, harmonic(ampsV, *popt), '--k')
print(f"I best fit amp: {0.5/popt[0]} V")

popt, pcov = curve_fit(harmonic, ampsV, Q, p0=p0)
print(popt)
plt.plot(ampsV, harmonic(ampsV, *popt), '--k')
print(f"Q best fit amp: {0.5/popt[0]} V")
