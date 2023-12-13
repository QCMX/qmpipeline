
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

filename = '{datetime}_qm_time_of_flight'
fpath = data_path(filename, datesuffix='_qm')

Navg = 20000

with qua.program() as tof_cal:
    n = qua.declare(int)  # variable for averaging loop
    adc_st = qua.declare_stream(adc_trace=True)  # stream to save ADC data
    n_st = qua.declare_stream()

    # Need to be on resonance when measuring transmission, so no changing here
    #update_frequency('resonator', 100e6)

    with qua.for_(n, 0, n < Navg, n + 1):
        qua.reset_phase('resonator') # reset the phase of the next played pulse
        #qua.play('preload', 'resonator')
        qua.measure('readout', 'resonator', adc_st)
        qua.wait(config.cooldown_clk, 'resonator')  # wait for photons in resonator to decay
        qua.save(n, n_st)

    with qua.stream_processing():
        n_st.save('iteration')
        adc_st.input1().average().save('adc1')
        adc_st.input2().average().save('adc2')

        # Will save only last run:
        adc_st.input1().save('adc1_single_run')
        adc_st.input2().save('adc2_single_run')


# #%%

# from qm import LoopbackInterface, SimulationConfig
# simulate_config = SimulationConfig(
#     duration=20000, # cycles
#     simulation_interface=LoopbackInterface(([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])))
# job = qmm.simulate(config.qmconfig, tof_cal, simulate_config)  # do simulation with qmm
# plt.figure()
# job.get_simulated_samples().con1.plot()  # visualize played pulses

# #%%
#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    # simulation properties
    from qm import SimulationConfig, LoopbackInterface
    simulate_config = SimulationConfig(
        duration=10000,
        simulation_interface=LoopbackInterface(([('con1', 1, 'con1', 1)])))
    job = qmm.simulate(config, tof_cal, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses
else:
    qm = qmm.open_qm(config.qmconfig)
    qminit.octave_setup_resonator(qm, config)
    tstart = time.time()
    try:
        
        job = qm.execute(tof_cal)  # execute QUA program
        res_handles = job.result_handles
        iteration_handle = res_handles.get('iteration')
        while res_handles.is_processing():
            iteration = (iteration_handle.fetch_all() or 0) + 1
            print(iteration)
            mpl_pause(1)
        #res_handles.wait_for_all_values()
    except KeyboardInterrupt as e:
        job.halt()
        raise e

    print(f"execution time: {time.time()-tstart:.1f}s")
    print(job.execution_report())

    adc1 = res_handles.get('adc1').fetch_all()
    adc2 = res_handles.get('adc2').fetch_all()
    adc1_single_run = res_handles.get('adc1_single_run').fetch_all()
    adc2_single_run = res_handles.get('adc2_single_run').fetch_all()

    np.savez_compressed(
        fpath, adc1=adc1, adc2=adc2, Navg=Navg,
        adc1_single_run=adc1_single_run, adc2_single_run=adc2_single_run,
        config=config.meta)

    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].set_title('Single run (Check ADCs saturation: |ADC|<2048)', fontsize=10)
    axs[0].plot(adc1_single_run, label='I')
    axs[0].plot(adc2_single_run, label='Q')
    axs[0].set_ylabel('ADC units')
    axs[0].legend()

    axs[1].set_title(f'Averaged run n={Navg} (Check ToF ({config.time_of_flight}ns) & DC Offset)', fontsize=10)
    axs[1].plot(adc1)
    axs[1].plot(adc2)
    axs[1].set_ylabel('ADC units')
    axs[-1].set_xlabel('sample (ns)')
    readoutpower = 10*np.log10(config.readout_amp**2 * 10) # V to dBm
    fig.suptitle(
        f"LO={config.resonatorLO/1e9:.5f}GHz   IF={config.resonatorIF/1e6:.3f}MHz"
        f"   Cooldown {config.cooldown_clk*4}ns   Navg {Navg}"
        f"\n{config.readout_len}ns readout at {readoutpower:.1f}dBm{config.resonator_output_gain:+.1f}dB"
        f",   {config.resonator_input_gain:+.1f}dB input gain"
        f"\n{config.qmconfig['elements']['resonator']['smearing']}ns smearing", fontsize=10)
    fig.tight_layout()
    fig.savefig(fpath+'.png')

    # np.savez('time_of_flight_single', adc1=adc1_single_run, adc2=adc2_single_run, config=config)
    # np.savez('time_of_flight_n=10000', adc1=adc1, adc2=adc2, config=config)

    print('mean', np.mean(adc1), np.mean(adc2), 'ADC units')
    print('mean', np.mean(adc1)/2**12, np.mean(adc2)/2**12, 'V')
    print('mean err', np.std(adc1)/np.sqrt(len(adc1))/2**12, np.std(adc2)/np.sqrt(len(adc1))/2**12, 'V')
    print('std', np.std(adc1)/2**12, np.std(adc2)/2**12, 'V')
    
    print("Correct with offset", config.adcoffset[0]-np.mean(adc1)/2**12, ",", config.adcoffset[1]-np.mean(adc2)/2**12)

    offsetbits = (config.adcoffset * 2**12).astype(int)
    amax = np.array([np.max(adc1_single_run), np.max(adc2_single_run)])
    amin = np.array([np.min(adc1_single_run), np.min(adc2_single_run)])
    if np.any(amax >= +2047+offsetbits-1) or np.any(amin <= -2048+offsetbits+1):
        print("WARNING: Some ADC values are saturating.")
    if np.any(amin < -2048+offsetbits) or np.any(amax > 2047+offsetbits):
        print("WARNING: Some ADC values are overflowing and wrapping.")

    print('ADC point noise:', np.std(adc1_single_run)/2**12, np.std(adc2_single_run)/2**12, 'V')

#%%

from uncertainties import ufloat
from scipy.optimize import curve_fit

# actually fIF doesn't need to be a fit parameter. could be fixed to config value
def cavity_load(t, fIF, lifetime, t0, amp, offs, phase):
    env = amp * (1 - np.exp(-(t-t0)/lifetime))
    env[t<t0] = 0
    return offs + env * np.sin(2*np.pi * t * fIF + phase)

t = np.arange(adc1.size) / 1e3 # us
p0 = [config.resonatorIF/1e6, 0.5, 250/1e3, 50, np.mean(adc1), 0]
popt, pcov = curve_fit(cavity_load, t, adc1, p0=p0)

res = [ufloat(opt, err) for opt, err in zip(popt, np.sqrt(np.diag(pcov)))]
for r, name, unit in zip(res, ["fIF", "lifet", "t0", "amp", "offs", "phase"], ["MHz", "us", "us", "ADC", "ADC", "rad"]):
    print(f"  {name:6s} {r} {unit}")

tsuper = np.linspace(t[0], t[-1], t.size*10)
fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(t, adc1, label='data', zorder=3)
#axs[0].plot(t, cavity_load(t, *p0), 'k--')
axs[0].plot(tsuper, cavity_load(tsuper, *popt), 'k-', linewidth=0.8, label='fit', zorder=2)
axs[0].axvline(popt[2], color='C2', label=f't0={res[2]*1e3}ns')
axs[0].axvline(popt[2]+popt[1], color='C3', label=f'lifet={res[1]}us')
axs[0].fill_between([t[0], t[-1]], [popt[4]-popt[3], popt[4]-popt[3]], [popt[4]+popt[3], popt[4]+popt[3]], alpha=0.1, zorder=-1)
axs[0].legend(fontsize=8)
axs[0].set_ylabel('ADC1 / ADC units')
axs[1].plot(t, adc1 - cavity_load(t, *popt))
axs[1].set_ylabel('residuals')
axs[1].set_xlabel('time / us')
fig.tight_layout()
fig.savefig(fpath+"_adc1fit.png")

print("ADC offset", res[4]/2**12, "V")

#%%

f = np.fft.rfftfreq(len(adc1_single_run), 1e-9)
fft = np.fft.rfft(adc1_single_run-np.mean(adc1_single_run))
plt.figure()
plt.plot(f/1e6, 20*np.log10(np.abs(fft)))
plt.xlabel("f/MHz")
plt.xlim(0,400)

#%%

# job.halt()
# qm.octave.set_rf_output_mode(element, octave.RFOutputMode.off)

