#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from helpers import data_path, mpl_pause, DurationEstimator

import configuration_pipeline as config_pipeline
import qminit, qmtools

qmm = qminit.connect()

#%%
from instruments.basel import BaselDACChannel

gate = BaselDACChannel(7) # 5 GHz

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 2e-6
GATERAMP_STEPTIME = 0.02

#%%

importlib.reload(config_pipeline)
importlib.reload(qminit)

# TODO make list programmatically and check against when choosing LO freq
# Note: requires re-opening the QM afterwards; persistent independent of config
mixercal = qmtools.QMMixerCalibration(qmm, config_pipeline, qubitLOs=[2e9, 2.1e9, 2.2e9, 2.3e9, 2.4e9, 2.5e9, 2.6e9, 2.7e9, 2.8e9, 2.9e9, 3.0e9, 3.1e9, 3.2e9, 3.3e9, 3.4e9, 3.5e9, 3.6e9, 3.7e9])

#%%
importlib.reload(config_pipeline)
importlib.reload(qminit)

filename = '{datetime}_qmpipeline_2tone'
fpath = data_path(filename, datesuffix='_qm')
Vgate = np.concatenate([np.linspace(-6.775, -6.815, 201)])
Vgate = np.concatenate([np.linspace(-5.18, -5.23, int(2.5e2)+1)])
#Vgate = np.array([-5.281])
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert len(Vgate) == 1 or Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

fr_range = np.arange(204.5e6, 209.5e6, 0.02e6) # IF Hz
fq_range = np.arange(-450e6, +450e6, 2e6)
fq_amps = np.logspace(np.log10(0.00316), np.log10(0.056), 11)
print(f"fq power range: {10*np.round(np.log10(fq_amps**2 * 10))[[0, -1]]} dBm")

# Typically see things in +- 100MHz around qubit, which should be just below IF=-100Mhz
# if estimated fq is good. Note: IF resolution is limited by pulse length.
# 10ns corresponds to 100Mhz resolution.
#time_rabi_IFs = np.arange(-400e6, -250e6, 30e6)
time_rabi_IFs = np.arange(-400e6, 50e6, 20e6)
time_rabi_max_duration = 72 # ns


# Vhyst = -5.20
# print(f"Gate hysteresis sweep ({abs(gate.get_voltage()-Vhyst)/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
# gate.ramp_voltage(Vhyst, 4*GATERAMP_STEP, GATERAMP_STEPTIME)

try:
    if vna.rf_output():
        print("Turning off VNA RF OUTPUT")
        vna.rf_output(False)
except:
    pass

print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)

baseconfig = qmtools.config_module_to_dict(config_pipeline)

# Not persistent.
# Need to redo after every config reload
print("Calibrate input ADC offset and get single shot noise")
localconfig = deepcopy(baseconfig)
localconfig['readout_amp'] = 0
pq = qmtools.QMTimeOfFlight(qmm, localconfig, Navg=500)
tof = pq.run()
pq.apply_offset_to(baseconfig)

barefreq = 5.05544e9
bareIF = barefreq - baseconfig['resonatorLO']
# def fq_estimate(deltafr):
#     """Convert resonance shift (Hz, compared to no qubit) to qubit frequency (Hz)"""
#     return 0.93 * 0.270e9 * (deltafr/1e3)**(0.328)
def fq_estimate(deltafr):
    """Convert resonance shift (Hz, compared to no qubit) to qubit frequency (Hz)"""
    return 0.270e9 * (deltafr/1e3)**(0.328)
GATE_SETTLETIME = 5 # s

# each name before colon indicates which program result is saved, i.e. the data format
# after the colon are identifiers if the same program is run multiple times (different settings)
results = {
    'mixer_calibration': [None]*Ngate,
    'resonator': [None]*Ngate,
    'resonator_P1': [None]*Ngate,
    'resonator_noise': [None]*Ngate,
    'resonator:hpr': [None]*Ngate,
    'qubit_P2': [None]*Ngate,
    'qubit_P2:zoom': [None]*Ngate,
    'qubit_three_tone:1': [None]*Ngate,
    'qubit_three_tone:2': [None]*Ngate,
    'readoutSNR': [None]*Ngate,
}
resonator = np.full((Vgate.size, fr_range.size), np.nan+0j)
frfit = np.full((Vgate.size, 4, 2), np.nan)

fig, axs = plt.subplots(nrows=2, ncols=5, layout='constrained', figsize=(16, 7))
frline, = axs[0,0].plot(Vgate, np.full(len(Vgate), np.nan), '.-', linewidth=1)
axs[0,0].set_xlabel('Vgate / V')
axs[0,0].set_ylabel('df / MHz')
axs[0,0].set_title('Cavity resonance vs gate')

progs = [None]*10
estimator = DurationEstimator(Ngate)
try:
    for i in range(Ngate):
        estimator.step(i) # at beginning to allow 'continue' during loop

        print(f"Setting gate to {Vgate[i]}V")
        gate.ramp_voltage(Vgate[i], GATERAMP_STEP, GATERAMP_STEPTIME)
        mpl_pause(GATE_SETTLETIME)

        for j in range(len(progs)):
            if progs[j] is not None:
                progs[j].clear_liveplot()
                progs[j] = None

        mixercal.run_after_interval(3*3600)
        results['mixer_calibration'][i] = True
        # Contains qm objects which we don't want to pickle
        #results['mixer_calibration'][i] = mixercal.run_after_interval(3*3600)

        localconfig = deepcopy(baseconfig)

        #######
        # One tone spectroscopy
        # localconfig['readout_amp'] = 0.0316 # -20dBm
        localconfig['qubit_output_gain'] = -15 # turn off qubit; -15dB is minimum for 2GHz

        prog = progs[0] = qmtools.QMResonatorSpec(qmm, localconfig, Navg=500, resonatorIFs=fr_range)
        results['resonator'][i] = resonatorspec = prog.run(plot=axs[1,0])
        resonator[i] = resonatorspec['Z']

        # Fit resonance, update resonator IF
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

        # # High power readout
        # # Shows error because can only scale up factor 2 the amplitude compared  to config
        # localconfig['readout_amp'] = 0.316
        # prog = progs[0] = qmtools.QMResonatorSpec(qmm, localconfig, Navg=500, resonatorIFs=fr_range)
        # results['resonator:hpr'][i] = resonatorspec = prog.run(plot=axs[1,0])
        # resonator[i] = resonatorspec['Z']
        # # Restore readout power
        # localconfig['readout_amp'] = baseconfig['readout_amp']

        #######
        # Resonator noise
        prog = progs[1] = qmtools.QMNoiseSpectrum(qmm, localconfig, Nsamples=100000, fcut_Hz=20e3)
        results['resonator_noise'][i] = prog.run(plot=axs[1,1])
        axs[1,1].set_xlim(-210,210)

        # #######
        # # Readout power
        # # Turn up octave output and not OPX
        # localconfig['resonator_output_gain'] = +10

        # # axs[0,1].axhline(
        # #     qmtools.opx_amp2pow(baseconfig['readout_amp'], baseconfig['resonator_output_gain']),
        # #     color='red', linestyle='--', linewidth=0.8, zorder=100)
        # # axs[0,1].plot(
        # #     [localconfig['resonatorIF']/1e6],
        # #     [qmtools.opx_amp2pow(baseconfig['short_readout_amp'], baseconfig['resonator_output_gain'])],
        # #     '.', color='r')
        # prog = progs[2] = qmtools.QMResonatorSpec_P2(
        #     qmm, localconfig, Navg=1000,
        #     resonatorIFs=np.arange(204e6, 209e6, 0.1e6),
        #     readout_amps=np.logspace(np.log10(0.0001), np.log10(0.0316), 26))
        # results['resonator_P1'][i] = prog.run(plot=axs[0,1])

        # # restore readout power
        # localconfig['readout_amp'] = baseconfig['readout_amp']
        # localconfig['resonator_output_gain'] = baseconfig['resonator_output_gain']

        #######
        # estimate qubit LO
        deltafr = max(localconfig['resonatorIF'] - bareIF, 1e2)
        fqest = fq_estimate(deltafr)
        print(f"  Resonator shift: {deltafr/1e6}MHz")
        print(f"  Estimated qubit freq: {fqest/1e9}GHz")
        if fqest < 1.4e9:
            print("  Skipping due to low estimated fq.")
            mpl_pause(0.1)
            continue
        qubitLO = int(max(2e9, np.ceil((fqest+0.10e9)/1e8)*1e8))
        # qubitLO = 2e9
        # if qubitLO == 3e9:
        #     qubitLO = 3.1e9
        print(f"  Choose qubit LO at {qubitLO/1e9}GHz")
        localconfig['qubitIF'] = fqest - qubitLO
        # localconfig['qmconfig']['mixers']['octave_octave1_2'][0]['intermediate_frequency'] = fqest - qubitLO
        localconfig['qubitLO'] = qubitLO
        localconfig['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency'] = qubitLO
        localconfig['qmconfig']['mixers']['octave_octave1_2'][0]['lo_frequency'] = qubitLO

        # high fq, fq = 2.5GHz, need lower power
        highfq = localconfig['resonatorIF'] > 206.7e6

        #######
        # 2tone spectroscopy qubit vs power & update qubit IF

        localconfig['qubit_output_gain'] = -10

        # Mark expected qubit freq in plot
        axs[0,2].axvline((fqest-qubitLO)/1e6, color='red', linestyle='--', linewidth=0.8, zorder=100)
        # # mark potential crosstalk
        # def inrange(a, v):
        #     return v > np.min(a) and v < np.max(a)
        # fr = localconfig['resonatorLO'] + resonatorfit[0][0]
        # if inrange(2*(qubitLO + fq_range), fr):
        #     axs[0,2].axvline((fr/2 - qubitLO)/1e6, color='gray', linewidth=0.8, zorder=99)
        # if inrange(2*(qubitLO - fq_range), fr):
        #     axs[0,2].axvline(-(fr/2 - qubitLO)/1e6, color='gray', linewidth=0.8, zorder=99)
        assert np.max(fq_amps) <= 0.3161 # max 0dBm
        prog = progs[3] = qmtools.QMQubitSpec_P2(
            qmm, localconfig, Navg=400, qubitIFs=fq_range,
            drive_amps=fq_amps * (0.178 if highfq else 1))
        results['qubit_P2'][i] = qubitspecvspower = prog.run(plot=axs[0,2])

        # find actual qubit freq, axes: freqs, amps
        try:
            fqIF = prog.find_dip(good_power_estimate_dBm=-55 if highfq else -40, ax=axs[0,2], apply_to_config=localconfig)
            fq = fqIF + qubitLO
        except qmtools.PipelineException as e:
            print(e)
            # print("Skipping to next gate point")
            # continue
            print("fq update from fq vs P2 failed. use initial estimate")
            fq = fqest
        if np.isnan(fq):
            print("fq from fq vs P2 failed (gave nan), use initial estimate")
            fq = fqest

        if fq < 1.5e9:
            print(f"  Skipping due to low estimated fq={fq}.")
            mpl_pause(0.1)
            continue
        if 2.0e9 < fq < 2.1e9:
            if fqest < 2e9:
                # print(f"  fq={fq} skip due to parisitic mode at 2.05GHz")
                # continue
                print("  fq on parasitic mode, keep initial estimate")
                fq = fqest
            else:
                #print("f  fq estimated on parasitic mode, move 150MHz above")
                #fq += 150e6
                pass
        if fqest > 2.8e9:
            print("  fq estimated large close to 3.4GHz parasitic mode. Keep initial guess")
            fq = fqest
        # if 3.3e9 < fq < 3.5e9:
        #     print(f"  fq={fq} skip due to parisitic mode at 3.40GHz")
        #     continue

        # print(f"Updating qubit LO freq for fq={fq}")
        # qubitLO = int(max(2e9, np.ceil((fq+0.15e9)/1e8)*1e8)) # -250 to -150 MHz

        # qubitLO = 2e9
        # if qubitLO == 3e9:
        #     qubitLO = 3.1e9

        print(f"  Choose qubit LO at {qubitLO/1e9}GHz")
        localconfig['qubitIF'] = fq - qubitLO
        # localconfig['qmconfig']['mixers']['octave_octave1_2'][0]['intermediate_frequency'] = fqest - qubitLO
        localconfig['qubitLO'] = qubitLO
        localconfig['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency'] = qubitLO
        localconfig['qmconfig']['mixers']['octave_octave1_2'][0]['lo_frequency'] = qubitLO

        ############
        # Three tone

        # localconfig['saturation_amp'] = 1e-3 # -50 dB
        # prog = progs[4] = qmtools.QMQubitSpecThreeTone(
        #     qmm, localconfig, Navg=20000, third_amp=localconfig['saturation_amp'],
        #     thirdIFs=np.arange(-400e6, 400e6, 10e6))
        # results['qubit_three_tone:1'][i] = prog.run(plot=axs[1,2])

        # localconfig['saturation_amp'] = 3.16e-3 # -40 dB
        # prog = progs[4] = qmtools.QMQubitSpecThreeTone(
        #     qmm, localconfig, Navg=20000, third_amp=localconfig['saturation_amp'],
        #     thirdIFs=np.arange(-400e6, 400e6, 10e6))
        # results['qubit_three_tone:2'][i] = prog.run(plot=axs[1,2])

        # ######
        # High res fq vs P2
        # (shares axis with ReadoutSNR)

        # prog = progs[4] = qmtools.QMQubitSpec_P2(
        #     qmm, localconfig, Navg=2000,
        #     qubitIFs=np.arange(-50e6, +50e6, 0.5e6) + fq-qubitLO,
        #     drive_amps=np.logspace(np.log10(0.00177), np.log10(0.0316), 11))
        # results['qubit_P2:zoom'][i] = qubitspecvspower = prog.run(plot=axs[1,2])

        # #######
        # # Readout SNR

        # localconfig['saturation_amp'] = 0.1
        # localconfig['qubit_output_gain'] = -10 #baseconfig['qubit_output_gain']
        # localconfig['resonator_output_gain'] = 0 # baseconfig['resonator_output_gain'] # restore

        # prog = progs[4] = qmtools.QMReadoutSNR(
        #     qmm, localconfig, Navg=1e4,
        #     resonatorIFs=np.arange(204.5e6, 208.2e6, 0.2e6),
        #     readout_amps=np.logspace(np.log10(0.000316), np.log10(0.316), 19), #np.logspace(np.log10(0.001), np.log10(0.316), 11),
        #     drive_len=100)
        # results['readoutSNR'][i] = prog.run(plot=axs[1,2])

        # # Optimize readout power and frequency
        # localconfig['short_readout_amp'] = 0.316 # baseconfig['short_readout_amp'] # restore
        # if fq < 2.9e9:
        #     # Best SNR at cavity frequency for lower qubit frequencies
        #     localconfig['resonatorIF'] = 205.5e6
        #     localconfig['resonator_output_gain'] = -10
        # else:
        #     localconfig['resonator_output_gain'] = -20

        # axs[1,2].plot(
        #     [localconfig['resonatorIF']/1e6],
        #     [qmtools.opx_amp2pow(baseconfig['short_readout_amp'], baseconfig['resonator_output_gain'])],
        #     '.', color='r')
finally:
    estimator.end()
    print("Saving data")
    time.sleep(1) # Let user see the "saving" message to avoid KeyboardInterrupt

    if 'tof' not in globals():
        tof = None
    np.savez_compressed(
        fpath, Vgate=Vgate, results=results, baseconfig=baseconfig,
        resonator=resonator, resonatorfit=frfit,
        tof_precalibration=tof,
        fq_estimate=fq_estimate, barefreq=barefreq,
        measurement_duration=estimator.elapsed())
    fig.savefig(fpath+'.png')
    print("Data saved.")
