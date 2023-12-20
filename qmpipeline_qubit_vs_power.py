#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import savgol_filter

from helpers import data_path, mpl_pause, DurationEstimator

import configuration as config
import qminit, qmtools

qmm = qminit.connect()

#%%
from instruments.basel import BaselDACChannel

gate = BaselDACChannel(3) # 5 GHz
#gate = BaselDACChannel(7) # 7 GHz

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 5e-6
GATERAMP_STEPTIME = 0.02

#%%

importlib.reload(config)
importlib.reload(qminit)

# TODO make list programmatically and check against when choosing LO freq
# Note: requires re-opening the QM afterwards; persistent independent of config
print("Calibrate mixers")
mixercal = qmtools.QMMixerCalibration(qmm, config, qubitLOs=[2e9, 2.1e9, 2.2e9, 2.3e9, 2.4e9, 2.45e9, 2.6e9, 2.7e9, 2.8e9, 2.9e9, 3.01e9, 3.1e9, 3.2e9, 3.3e9, 3.4e9, 3.5e9, 3.6e9, 3.7e9])

#%%
importlib.reload(config)
importlib.reload(qminit)

filename = '{datetime}_qmpipeline'
fpath = data_path(filename, datesuffix='_qm')

#Vgate = np.concatenate([np.linspace(-4.92, -4.93, 401)])
# Vgate = np.concatenate([np.linspace(-4.346, -4.335, 21)])
# Vgate = np.concatenate([np.linspace(-4.341, -4.337, 21)])
Vgate = np.concatenate([np.linspace(-4.42, -4.40, 201)])
Vgate = np.array([-4.415])
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert len(Vgate) == 1 or Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

fr_range = np.arange(204.5e6, 209.5e6, 0.02e6) # IF Hz
fq_range = np.arange(-400e6, +400e6, 5e6)
fq_amps = np.logspace(np.log10(0.01), np.log10(0.316), 15) # -30 to 0
print(f"fq power range: {10*np.round(np.log10(fq_amps**2 * 10))[[0, -1]]} dBm")

# Typically see things in +- 100MHz around qubit, which should be just below IF=-100Mhz
# if estimated fq is good. Note: IF resolution is limited by pulse length.
# 10ns corresponds to 100Mhz resolution.
time_rabi_IFs = np.arange(-340e6, -80e6, 20e6)
time_rabi_max_duration = 40 # ns

print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)

baseconfig = qmtools.config_module_to_dict(config)

# # Not persistent.
# # Need to redo after every config reload
# print("Calibrate input ADC offset")
# localconfig = deepcopy(baseconfig)
# localconfig['readout_amp'] = 0
# pq = qmtools.QMTimeOfFlight(qmm, localconfig, Navg=500)
# tof = pq.run()
# pq.apply_offset_to(baseconfig)

barefreq = 5.05578e9
bareIF = barefreq - baseconfig['resonatorLO']
def fq_estimate(deltafr):
    """Convert resonance shift (Hz, compared to no qubit) to qubit frequency (Hz)"""
    return 0.375e9 * (deltafr/1e3)**(0.285)
GATE_SETTLETIME = 5

results = {
    'mixer_calibration': [None]*Ngate,
    'resonator': [None]*Ngate,
    'resonator_P1': [None]*Ngate,
    'qubit_P2': [None]*Ngate,
    'readoutSNR': [None]*Ngate,
    'time_rabi': [None]*Ngate,
    'time_rabi_chevrons': [None]*Ngate,
    'power_rabi': [None]*Ngate,
}
resonator = np.full((Vgate.size, fr_range.size), np.nan+0j)
frfit = np.full((Vgate.size, 4, 2), np.nan)

fig, axs = plt.subplots(nrows=2, ncols=4, layout='constrained', figsize=(12, 6))
frline, = axs[0,0].plot(Vgate, np.full(len(Vgate), np.nan), '.-', linewidth=1)
axs[0,0].set_xlabel('Vgate / V')
axs[0,0].set_ylabel('df / MHz')
axs[0,0].set_title('Cavity resonance vs gate')

progs = [None, None, None, None, None, None]
estimator = DurationEstimator(Ngate)
try:
    for i in range(Ngate):
        print(f"Setting gate to {Vgate[i]}V")
        gate.ramp_voltage(Vgate[i], GATERAMP_STEP, GATERAMP_STEPTIME)
        mpl_pause(GATE_SETTLETIME)
        for j in range(len(progs)):
            if progs[j] is not None:
                progs[j].clear_liveplot()
                progs[j] = None

        results['mixer_calibration'][i] = mixercal.run_after_interval(3*3600)

        localconfig = deepcopy(baseconfig)

        # One tone spectroscopy
        localconfig['readout_amp'] = 0.0316 # -20dBm
        localconfig['qubit_output_gain'] = -15 # minimum for 2GHz
        prog = progs[0] = qmtools.QMResonatorSpec(qmm, localconfig, Navg=500, resonatorIFs=fr_range)
        resonatorspec = prog.run(plot=axs[1,0])
        results['resonator'][i] = resonatorspec
        resonator[i] = resonatorspec['Z']

        # Fit resonance
        resonatorfit = prog.fit_lorentzian(ax=axs[1,0])
        frfit[i,:,0], frfit[i,:,1] = resonatorfit
        # show in Vgate plot
        frline.set_ydata((frfit[:,0,0] - bareIF)/1e6)
        axs[0,0].relim(), axs[0,0].autoscale(), axs[0,0].autoscale_view()

        if resonatorfit[0][1] < 100e3 or resonatorfit[0][1] >  3e6:
            print(f"  Resonator width {resonatorfit[0][1]/1e3}kHz out of good range. Skip.")
            mpl_pause(0.1)
            continue
        localconfig['resonatorIF'] = resonatorfit[0][0]

        # # Readout power
        # # Turn up octave output and not OPX
        # axs[0,1].axhline(
        #     qmtools.opx_amp2pow(baseconfig['readout_amp'], baseconfig['resonator_output_gain']),
        #     color='fuchsia', linewidth=0.8, zorder=100)
        # localconfig['resonator_output_gain'] = 10
        # prog = progs[1] = qmtools.QMResonatorSpec_P2(
        #     qmm, localconfig, Navg=500,
        #     resonatorIFs=np.arange(204e6, 209e6, 0.1e6),
        #     readout_amps=np.logspace(np.log10(0.0001), np.log10(0.0316), 21))
        # resonatorP1 = prog.run(plot=axs[0,1])
        # results['resonator_P1'][i] = resonatorP1
        # localconfig['resonator_output_gain'] = baseconfig['resonator_output_gain'] # restore

        # determine qubit LO
        deltafr = localconfig['resonatorIF'] - bareIF
        fqest = fq_estimate(deltafr)
        print(f"  Resonator shift: {deltafr/1e6}MHz")
        print(f"  Estimated qubit freq: {fqest/1e9}GHz")
        if fqest < 1.7e9:
            print("  Skipping due to low estimated fq.")
            mpl_pause(0.1)
            continue
        # qubitLO = int(max(2e9, np.floor((fqest+0.350e9)/1e8)*1e8))
        qubitLO = int(max(2e9, np.ceil((fqest+0.20e9)/1e8)*1e8))
        if qubitLO == 2.5e9: # avoid 2*LO close to resonator
            qubitLO = 2.45e9
        if qubitLO == 3e9:
            qubitLO = 3.1e9
        print(f"  Choose qubit LO at {qubitLO/1e9}GHz")
        localconfig['qubitIF'] = fqest - qubitLO
        # localconfig['qmconfig']['mixers']['octave_octave1_2'][0]['intermediate_frequency'] = fqest - qubitLO
        localconfig['qubitLO'] = qubitLO
        localconfig['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency'] = qubitLO
        localconfig['qmconfig']['mixers']['octave_octave1_2'][0]['lo_frequency'] = qubitLO

        # Mark expected qubit freq in plot
        axs[1,1].axvline((fqest-qubitLO)/1e6, color='fuchsia', linewidth=0.8, zorder=100)
        # mark potential crosstalk
        def inrange(a, v):
            return v > np.min(a) and v < np.max(a)
        fr = localconfig['resonatorLO'] + resonatorfit[0][0]
        if inrange(2*(qubitLO + fq_range), fr):
            axs[1,1].axvline((fr/2 - qubitLO)/1e6, color='gray', linewidth=0.8, zorder=99)
        if inrange(2*(qubitLO - fq_range), fr):
            axs[1,1].axvline(-(fr/2 - qubitLO)/1e6, color='gray', linewidth=0.8, zorder=99)

        # qubit vs power
        assert np.max(fq_amps) <= 0.3161 # max 0dBm
        localconfig['qubit_output_gain'] = +5
        prog = progs[2] = qmtools.QMQubitSpec_P2(
            qmm, localconfig, Navg=600, qubitIFs=fq_range, drive_amps=fq_amps)
        qubitspecvspower = prog.run(plot=axs[1,1])
        results['qubit_P2'][i] = qubitspecvspower

        # find actual qubit freq, axes: freqs, amps
        try:
            fqIF = prog.find_dip(good_power_estimate_dBm=-10, ax=axs[1,1], apply_to_config=localconfig)
            fq = fqIF + qubitLO
        except qmtools.PipelineException as e:
            print(e)
            # print("Continuing with estimated qubit IF")
            continue

        print("Updating qubit LO freq to")
        if fq < 2.4e9:
            print(f"  {fq} <2.4GHz, skip due to parisitic mode")
            continue
        if fq > 3.3e9:
            print(f"  {fq} >3.3 GHz, skip due to parisitic mode")
            continue
        if fqest < 1.7e9:
            print("  Skipping due to low estimated fq.")
            mpl_pause(0.1)
            continue
        # qubitLO = int(max(2e9, np.floor((fqest+0.350e9)/1e8)*1e8))
        qubitLO = int(max(2e9, np.ceil((fq+0.20e9)/1e8)*1e8)) # -300 to -200 MHz
        if qubitLO == 2.5e9: # avoid 2*LO close to resonator
            qubitLO = 2.45e9
        if qubitLO == 3e9:
            qubitLO = 3.1e9
        print(f"  Choose qubit LO at {qubitLO/1e9}GHz")
        localconfig['qubitIF'] = fq - qubitLO
        # localconfig['qmconfig']['mixers']['octave_octave1_2'][0]['intermediate_frequency'] = fqest - qubitLO
        localconfig['qubitLO'] = qubitLO
        localconfig['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency'] = qubitLO
        localconfig['qmconfig']['mixers']['octave_octave1_2'][0]['lo_frequency'] = qubitLO

        # # Readout SNR (shares axes with readout vs power)
        # axs[0,1].axhline(
        #     qmtools.opx_amp2pow(baseconfig['short_readout_amp'], baseconfig['resonator_output_gain']),
        #     color='fuchsia', linewidth=0.8, zorder=100)
        # localconfig['saturation_amp'] = fq_amps[ampi]
        # localconfig['resonator_output_gain'] = 10
        # prog = progs[1] = qmtools.QMReadoutSNR(
        #     qmm, localconfig, Navg=20e3,
        #     resonatorIFs=np.arange(190e6, 230e6, 2e6),
        #     readout_amps=np.logspace(np.log10(0.000316), np.log10(0.316), 21),
        #     drive_len=localconfig['short_readout_len']+60)
        # results['readoutSNR'][i] = prog.run(plot=axs[0,1])
        # localconfig['saturation_amp'] = baseconfig['saturation_amp'] # restore
        # localconfig['resonator_output_gain'] = baseconfig['resonator_output_gain'] # restore

        localconfig['saturation_amp'] = 0.316 # 0dBm
        localconfig['qubit_output_gain'] = +5
        localconfig['resonator_output_gain'] = +5

        # Time Rabi
        prog = progs[3] = qmtools.QMTimeRabi(
            qmm, localconfig, Navg=2e6,
            max_duration_ns=time_rabi_max_duration,
            drive_read_overlap_cycles=0)
        # if i == 0:
        #     prog.check_timing(plot=True)
        results['time_rabi'][i] = prog.run(plot=axs[0,2])

        # # Time Rabi Chevrons
        # prog = progs[4] = qmtools.QMTimeRabiChevrons(
        #     qmm, localconfig, Navg=1e6,
        #     qubitIFs=time_rabi_IFs,
        #     max_duration_ns=time_rabi_max_duration,
        #     drive_read_overlap_cycles=0)
        # results['time_rabi_chevrons'][i] = prog.run(plot=axs[1,2])

        # Power Rabi
        prog = progs[5] = qmtools.QMPowerRabi(
            qmm, localconfig, Navg=2e6, duration_ns=8,
            drive_amps=np.linspace(0, 0.316, 15),
            drive_read_overlap_cycles=0)
        # if i == 0:
        #     prog.check_timing(plot=True)
        results['power_rabi'][i] = prog.run(plot=axs[0,3])

        # Relaxation

        estimator.step(i)
finally:
    print("Saving data")
    if 'tof' not in globals():
        tof = None
    np.savez_compressed(
        fpath, Vgate=Vgate, results=results, baseconfig=baseconfig,
        resonator=resonator, resonatorfit=frfit,
        tof_precalibration=tof,
        measurement_duration=estimator.elapsed())
    fig.savefig(fpath+'.png')
