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
from qualang_tools.loops import from_array
from qualang_tools.bakery import baking

from helpers import data_path, mpl_pause
from qm_helpers import int_forloop_values

import configuration as config
import qminit

qmm = qminit.connect()

#%%
importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_power_rabi_highres'
fpath = data_path(filename, datesuffix='_qm')

Navg = int(1e6)

maxduration = 25 # cycles

# minimum wait: 4cycles = 16ns
drive_read_overlap = 4 # cycles
actual_overlap = drive_read_overlap + 4 # due to additional short waveform

# minimum duration: 4cycles = 16ns
duration_clk = np.arange(4, maxduration, 1)
duration_ns = np.arange(0, maxduration*4, 1)

try:
    Vgate = gate.get_voltage()
except:
    Vgate = np.nan

# Waveforms are padded to a multiple of 4 samples and a minimum length of 16 samples
# (with padding added as zeros at the beginning).
with baking(config.qmconfig, padding_method='left') as bake0:
    bake0.add_op('drive_0', 'qubit', [[0], [0]])
    bake0.play('drive_0', 'qubit')
baked_saturation = [bake0]
for l in range(1, 16):
    with baking(config.qmconfig, padding_method='left') as bake:
        bake.add_op('drive_%d'%l, 'qubit', [[config.saturation_amp]*l, [0]*l])
        bake.play('drive_%d'%l, 'qubit')
    baked_saturation.append(bake)

with qua.program() as time_rabi:
    n = qua.declare(int)
    t = qua.declare(int)
    I = qua.declare(qua.fixed)
    Q = qua.declare(qua.fixed)
    n_st = qua.declare_stream()
    I_st = qua.declare_stream()
    Q_st = qua.declare_stream()
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Navg, n + 1):
        for l in range(16):
            qua.align()
            if drive_read_overlap:
                qua.wait(drive_read_overlap, 'qubit')
            baked_saturation[l].run()
            qua.measure('short_readout', 'resonator', None,
                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
            qua.wait(config.cooldown_clk, 'resonator')
            qua.wait(rand.rand_int(50)+4, 'resonator') # randomize demod error

        with qua.for_(*from_array(t, duration_clk)):
            if drive_read_overlap:
                qua.wait(drive_read_overlap, 'qubit')
            baked_saturation[0].run()
            qua.play('saturation', 'qubit', duration=t)
            qua.wait(t, 'resonator')
            qua.measure('short_readout', 'resonator', None,
                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
            qua.wait(config.cooldown_clk, 'resonator')
            qua.wait(rand.rand_int(50)+4, 'resonator')
            
            qua.align()
            if drive_read_overlap:
                qua.wait(drive_read_overlap, 'qubit')
            baked_saturation[1].run()
            qua.play('saturation', 'qubit', duration=t)
            qua.wait(t, 'resonator')
            qua.measure('short_readout', 'resonator', None,
                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
            qua.wait(config.cooldown_clk, 'resonator')
            qua.wait(rand.rand_int(50)+4, 'resonator')
            
            qua.align()
            if drive_read_overlap:
                qua.wait(drive_read_overlap, 'qubit')
            baked_saturation[2].run()
            qua.play('saturation', 'qubit', duration=t)
            qua.wait(t, 'resonator')
            qua.measure('short_readout', 'resonator', None,
                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
            qua.wait(config.cooldown_clk, 'resonator')
            qua.wait(rand.rand_int(50)+4, 'resonator')
            
            qua.align()
            if drive_read_overlap:
                qua.wait(drive_read_overlap, 'qubit')
            baked_saturation[3].run()
            qua.play('saturation', 'qubit', duration=t)
            qua.wait(t, 'resonator')
            qua.measure('short_readout', 'resonator', None,
                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
            qua.wait(config.cooldown_clk, 'resonator')
            qua.wait(rand.rand_int(50)+4, 'resonator')
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(duration_clk)*4+16).average().save('I')
        Q_st.buffer(len(duration_clk)*4+16).average().save('Q')

# #%%
# from qm import LoopbackInterface, SimulationConfig
# simulate_config = SimulationConfig(
#     duration=20000, # cycles
#     simulation_interface=LoopbackInterface(([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])))
# job = qmm.simulate(config.qmconfig, time_rabi, simulate_config)  # do simulation with qmm

# plt.figure()
# job.get_simulated_samples().con1.plot()  # visualize played pulses
# #plt.savefig(fpath+"_pulses.png")
# #%%

qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config, short_readout_gain=True)
qminit.octave_setup_qubit(qm, config)

job = qm.execute(time_rabi)  # execute QUA program
tstart = time.time()

res_handles = job.result_handles  # get access to handles
I_handle = res_handles.get('I')
I_handle.wait_for_values(1)
Q_handle = res_handles.get('Q')
Q_handle.wait_for_values(1)
iteration_handle = res_handles.get('iteration')
iteration_handle.wait_for_values(1)

fig, ax = plt.subplots()
ax.set_xlabel('drive duration / ns')
ax.set_ylabel('arg (demod signal)')
ax.set_title('time rabi')
line, = ax.plot(duration_ns, np.full(duration_ns.size, np.nan))
fig.tight_layout()
try:
    while res_handles.is_processing():
        iteration = iteration_handle.fetch_all() + 1
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        Z = I + Q*1j
        line.set_ydata(np.angle(Z))
        ax.set_title(f'time rabi {iteration}/{Navg}')
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

trun = time.time() - tstart
res_handles.wait_for_all_values()
print(f"execution time: {trun:.1f}s")
print(job.execution_report())

Nactual = iteration_handle.fetch_all()+1
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
Zraw = I + 1j * Q

np.savez_compressed(
    fpath, Zraw=Zraw, Navg=Navg, Nactual=Nactual,
    duration_clk=duration_clk, duration_ns=duration_ns,
    drive_read_overlap_clk=actual_overlap,
    Vgate=Vgate, config=config.meta)

plt.close(fig)

Zcorr = Zraw / config.short_readout_len * 2**12

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(duration_ns, 10*np.log10((np.abs(Zcorr))**2 * 10))
ax2.plot(duration_ns, np.unwrap(np.angle(Zcorr)))
ax1.grid(), ax2.grid()
ax1.set_ylabel('|S| / dB')
ax2.set_ylabel('arg S / rad')
ax2.set_xlabel('drive duration / ns')
readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
drivepower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
fig.suptitle(
    f"resonator {(config.resonatorLO+config.resonatorIF)/1e9:f}GHz"
    f"   qubit {(config.qubitLO+config.qubitIF)/1e9:f}GHz"
    f"   Navg {Nactual}"
    f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
    f"\ndrive pulse at {drivepower:.1f}dBm{config.qubit_output_gain:+.1f}dB   waveform overlap {actual_overlap*4:d}ns"
    f"\nruntime {trun:.1f}s   Vgate={Vgate:.6f}V",
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
