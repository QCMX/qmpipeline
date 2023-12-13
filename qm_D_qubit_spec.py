
# Before running this need to
# - calibrate mixers (calibration db)
# - set time of flight (config.time_of_flight)
# - set input adc offsets (config.adcoffset)
# - electrical delay correction (config.PHASE_CORR)
# - resonator IF, readout power

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import importlib
import qm.qua as qua

from helpers import data_path, mpl_pause

import configuration as config
import qminit

qmm = qminit.connect()

#%%
# Config depends on experiment cabling
# OPX imported from config in extra file, Octave config here. :/
# Octave config is persistent even when opening new qm

importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qm_qubit_spec'
fpath = data_path(filename, datesuffix='_qm')

Navg = 20000

f_min = 10e6
f_max = 550e6
df = 2e6
freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs

try:
    Vgate = gate.get_voltage()
except:
    Vgate = None

# Align readout pulse in middle of saturation pulse
#assert config.saturation_len > config.short_readout_len
readoutwait = 4# int(((config.saturation_len - config.short_readout_len) / 2) / 4) # cycles
print("Readoutwait", readoutwait*4, "ns")

with qua.program() as qubit_spec:
    n = qua.declare(int)  # variable for average loop
    n_st = qua.declare_stream()  # stream for 'n'
    f = qua.declare(int)  # variable to sweep freqs
    I = qua.declare(qua.fixed)  # demodulated and integrated signal
    Q = qua.declare(qua.fixed)  # demodulated and integrated signal
    I_st = qua.declare_stream()  # stream for I
    Q_st = qua.declare_stream()  # stream for Q
    rand = qua.lib.Random()

    with qua.for_(n, 0, n < Navg, n + 1):
        with qua.for_(f, f_min, f <= f_max, f + df):
            qua.update_frequency('qubit', f)
            # qua.reset_phase('resonator')
            qua.wait(config.cooldown_clk, 'resonator')
            qua.wait(rand.rand_int(50)+4, 'resonator')
            qua.align()
            qua.play('saturation', 'qubit')
            qua.wait(readoutwait, 'resonator')
            qua.measure('short_readout', 'resonator', None,
                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
            qua.save(I, I_st)
            qua.save(Q, Q_st)
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(freqs)).average().save('I')
        Q_st.buffer(len(freqs)).average().save('Q')

# #%%
# from qm import LoopbackInterface, SimulationConfig
# simulate_config = SimulationConfig(
#     duration=50000, # cycles
#     simulation_interface=LoopbackInterface(([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])))
# job = qmm.simulate(config.qmconfig, qubit_spec, simulate_config)  # do simulation with qmm

# plt.figure()
# job.get_simulated_samples().con1.plot()  # visualize played pulses
# #%%

# Execute program
qm = qmm.open_qm(config.qmconfig)
qminit.octave_setup_resonator(qm, config, short_readout_gain=True)
qminit.octave_setup_qubit(qm, config)
tstart = time.time()
job = qm.execute(qubit_spec)

res_handles = job.result_handles  # get access to handles
I_handle = res_handles.get('I')
I_handle.wait_for_values(1)
Q_handle = res_handles.get('Q')
Q_handle.wait_for_values(1)
iteration_handle = res_handles.get('iteration')
iteration_handle.wait_for_values(1)

# Live plotting
fig, ax = plt.subplots()
ax.set_xlabel('IF [Hz]')
ax.set_ylabel('arg (demod signal)')
ax.set_title('qubit spectroscopy analysis')
line, = ax.plot(freqs, np.full(freqs.size, np.nan))
fig.tight_layout()
try:
    while res_handles.is_processing():
        iteration = iteration_handle.fetch_all() + 1
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        Z = I + Q*1j
        line.set_ydata(np.angle(Z))
        ax.set_title(f'qubit spectroscopy analysis {iteration}/{Navg}')
        ax.relim()
        ax.autoscale()
        ax.autoscale_view()
        print(iteration)
        mpl_pause(0.5)
except KeyboardInterrupt:
    job.halt()
finally:
    print("Execution time", time.time()-tstart, "s")
    # Get and save data
    Nactual = iteration_handle.fetch_all()+1
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    Zraw = I + 1j * Q

    np.savez_compressed(
        fpath, Navg=Navg, Nactual=Nactual, freqs=freqs, Zraw=Zraw,
        Vgate=Vgate,
        config=config.meta)

    # Final plot
    plt.close(fig)
    
    Zcorr = Zraw / config.short_readout_len * 2**12
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(freqs/1e6, 10*np.log10((np.abs(Zcorr))**2 * 10))
    ax2.plot(freqs/1e6, np.unwrap(np.angle(Zcorr)))
    ax1.grid(), ax2.grid()
    ax1.set_ylabel('|S| / dB')
    ax2.set_xlabel('Qubit IF / MHz')
    ax2.set_ylabel('Phase / rad')
    readoutpower = 10*np.log10(config.short_readout_amp**2 * 10) # V to dBm
    saturationpower = 10*np.log10(config.saturation_amp**2 * 10) # V to dBm
    fig.suptitle(
        f"resonator {(config.resonatorLO+config.resonatorIF)/1e9:f}GHz"
        f"   qubit LO {config.qubitLO/1e9:.3f}GHz"
        f"   Navg {Nactual}"
        f"\n{config.short_readout_len}ns short readout at {readoutpower:.1f}dBm{config.resonator_output_gain+config.short_readout_amp_gain:+.1f}dB"
        f"\n{config.saturation_len}ns saturation pulse at {saturationpower:.1f}dBm{config.qubit_output_gain:+.1f}dB"
        f"\nVgate={Vgate}V",
        fontsize=8)
    fig.tight_layout()
    fig.savefig(fpath+'.png')
#%%

plt.figure()
plt.plot(freqs/1e6, np.abs(Zraw))
