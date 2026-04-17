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

importlib.reload(qmtools)

qmm = qminit.connect()
AMP0dBm = 10**(-5/10) # output in V to have 0dBm ADC

#%%
from instruments.basel import BaselDACChannel

gate = BaselDACChannel(7) # 5 GHz

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 2e-6
GATERAMP_STEPTIME = 0.02

#%% JPA pump

from RsInstrument import RsInstrument
jpapump = RsInstrument('TCPIP::169.254.2.22::INSTR', id_query=True, reset=False)


#%% JPA flux

from qcodes.instrument_drivers.yokogawa.GS200 import GS200

try:
    fluxbias.close()
    pass
except: pass

fluxbias = GS200("source", 'TCPIP0::169.254.0.2::inst0::INSTR', terminator="\n")
assert fluxbias.source_mode() == 'CURR'
assert fluxbias.output() == 'on'

#%%

importlib.reload(config_pipeline)
importlib.reload(qminit)

# TODO make list programmatically and check against when choosing LO freq
# Note: requires re-opening the QM afterwards; persistent independent of config
mixercal = qmtools.QMMixerCalibration(qmm, config_pipeline, qubitLOs=[
    2.0e9, 2.1e9, 2.2e9, 2.3e9, 2.4e9, 2.5e9, 2.6e9, 2.7e9, 2.8e9, 2.9e9,
    3.0e9, 3.1e9, 3.2e9, 3.3e9, 3.4e9, 3.5e9, 3.6e9, 3.7e9, 3.8e9, 3.9e9,
    4.0e9, 4.1e9])

#%%
importlib.reload(config_pipeline)
importlib.reload(qminit)

filename = '{datetime}_qmpipeline'
fpath = data_path(filename, datesuffix='_qm')
# Vgate = np.linspace(-3.97, -4.01, 11)
Vgate = np.linspace(-3.97, -3.985, 11)
# Vgate = np.array([-3.9764]) # funny Ramsey shape
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert len(Vgate) == 1 or Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

fr_range = np.arange(200e6, 212e6, 0.05e6) # IF Hz

# Length / distance of pulses for Rabi / Ramsey / Relaxation
# It's nice to have the same length for everything to plot cohesive results later
PROTOCOL_DURATION = 160

# Vhyst = -5.20
# print(f"Gate hysteresis sweep ({abs(gate.get_voltage()-Vhyst)/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
# gate.ramp_voltage(Vhyst, 4*GATERAMP_STEP, GATERAMP_STEPTIME)

try:
    if vna.rf_output():
        print("Turning off VNA RF OUTPUT")
        vna.rf_output(False)
except:
    pass

jpameta = {
    # 'fluxbias': fluxbias.current(), # A
    'fpump': float(jpapump.query(':source:freq?')),
    'Ppump': float(jpapump.query(':source:power?')),
    'output': jpapump.query_int(':output?') == 1
}
if not jpameta['output']:
    print("Info: JPA off [press enter]")
    input()

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

barefreq = 5.010e9 + 201.47e6
bareIF = barefreq - baseconfig['resonatorLO']
def fq_estimate(deltafr):
    """Convert resonance shift (Hz, compared to no qubit) to qubit frequency (Hz)"""
    # Inverse of deltafr = g**2 * EC / (delta * (delta - EC)), delta = fq - fr
    EC = 0.16e9
    g = 0.3e9
    fr = 5e9
    return -np.sqrt(EC)*np.sqrt(EC*deltafr + 4*g**2)/(2*np.sqrt(deltafr)) + EC/2 + fr

GATE_SETTLETIME = 5 # s

# each name before colon indicates which program result is saved, i.e. the data format
# after the colon are identifiers if the same program is run multiple times (different settings)
results = {
    'mixer_calibration': [None]*Ngate,
    'resonator': [None]*Ngate,
    'resonator_P1': [None]*Ngate,
    'resonator_noise': [None]*Ngate,
    'qubit_P2': [None]*Ngate,
    'readoutSNR': [None]*Ngate,
    'readoutSNR_P1': [None]*Ngate,
    'qubit:1': [None]*Ngate,
    'qubit:1hp': [None]*Ngate,
    'qubit:2': [None]*Ngate,
    'time_rabi:1': [None]*Ngate,
    'time_rabi:2': [None]*Ngate,
    'time_rabi:3': [None]*Ngate,
    'time_rabi_chevrons': [None]*Ngate,
    'time_rabi_chevrons:hpr': [None]*Ngate,
    'time_rabi_chevrons:inneravg': [None]*Ngate,
    'power_rabi:square_8ns': [None]*Ngate,
    'power_rabi:square_16ns': [None]*Ngate,
    'power_rabi:gaussian_8ns': [None]*Ngate,
    'power_rabi:gaussian_16ns': [None]*Ngate,
    'relaxation': [None]*Ngate,
    'relaxation:hpr': [None]*Ngate,
    'ramsey:on_resonance': [None]*Ngate,
    'ramsey:detuned': [None]*Ngate,
    'ramsey:detuned_repetition': [None]*Ngate,
    'ramsey_chevron_repeat:gaussian': [None]*Ngate,
    'ramsey_chevron_repeat:square': [None]*Ngate,
}
resonator = np.full((Vgate.size, fr_range.size), np.nan+0j)
frfit = np.full((Vgate.size, 4, 2), np.nan)

fig, axs = plt.subplots(nrows=2, ncols=7, layout='constrained', figsize=(16, 7))
frline, = axs[0,0].plot(Vgate, np.full(len(Vgate), np.nan), '.-', linewidth=1)
axs[0,0].set_xlabel('Vgate / V')
axs[0,0].set_ylabel('df / MHz')
axs[0,0].set_title('Cavity resonance vs gate')

progs = [None]*12
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

        mixercal.run_after_interval(6*3600)
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

        #######
        # Resonator noise
        prog = progs[1] = qmtools.QMNoiseSpectrum(qmm, localconfig, Nsamples=100000, fcut_Hz=20e3)
        results['resonator_noise'][i] = prog.run(plot=axs[1,1])
        axs[1,1].set_xlim(-210,210)

        # #######
        # # Readout power
        # # Turn up octave output and not OPX
        # localconfig['resonator_output_gain'] = -20
        # localconfig['qmconfig']['waveforms']['readout_wf']['sample'] = 0.316

        # prog = progs[2] = qmtools.QMResonatorSpec_P2(
        #     qmm, localconfig, Navg=100,
        #     resonatorIFs=fr_range,
        #     readout_amps=np.logspace(np.log10(0.00316), np.log10(0.316), 29))
        # results['resonator_P1'][i] = prog.run(plot=axs[0,1])

        # # restore readout power
        # localconfig['qmconfig']['waveforms']['readout_wf']['sample'] = baseconfig['qmconfig']['waveforms']['readout_wf']['sample']
        # localconfig['readout_amp'] = baseconfig['readout_amp']
        # localconfig['resonator_output_gain'] = baseconfig['resonator_output_gain']

        #######
        # estimate qubit LO
        deltafr = localconfig['resonatorIF'] - bareIF
        fqest = fq_estimate(deltafr)
        print(f"  Resonator shift: {deltafr/1e6}MHz")
        print(f"  Estimated qubit freq: {fqest/1e9}GHz")
        qubitLO = int(max(2e9, np.ceil((fqest+0.10e9)/1e8)*1e8))
        print(f"  Choose qubit LO at {qubitLO/1e9}GHz")
        localconfig['qubitIF'] = fqest - qubitLO
        localconfig['qubitLO'] = qubitLO
        localconfig['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency'] = qubitLO
        localconfig['qmconfig']['elements']['qubit2']['mixInputs']['lo_frequency'] = qubitLO

        #######
        # 2tone spectroscopy qubit vs power & update qubit IF
        # Note: have room-T amp, max input is +4dBm

        localconfig['qubit_output_gain'] = -15

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
        fq_amps = np.logspace(-40/20, 0, 9) * AMP0dBm
        assert np.max(fq_amps) <= 0.32 # max 0dBm
        prog = progs[3] = qmtools.QMQubitSpec_P2(
            qmm, localconfig, Navg=300, qubitIFs=np.arange(-450e6, +450e6, 2e6),
            drive_amps=fq_amps)
        results['qubit_P2'][i] = qubitspecvspower = prog.run(plot=axs[0,2])

        # # find actual qubit freq, axes: freqs, amps
        # try:
        #     fqIF = prog.find_dip(good_power_estimate_dBm=-40, ax=axs[0,2], apply_to_config=localconfig)
        #     fq = fqIF + qubitLO
        # except qmtools.PipelineException as e:
        #     print(e)
        #     # print("Skipping to next gate point")
        #     # continue
        #     print("fq update from fq vs P2 failed. use initial estimate")
        #     fq = fqest
        # if np.isnan(fq):
        #     print("fq from fq vs P2 failed (gave nan), use initial estimate")
        #     fq = fqest

        # if fq < 1.3e9:
        #     print(f"  Skipping due to low estimated fq={fq}.")
        #     mpl_pause(0.1)
        #     continue
        # print(f"Updating qubit LO freq for fq={fq}")
        # qubitLO = int(max(2e9, np.ceil((fq+0.15e9)/1e8)*1e8)) # -250 to -150 MHz
        # print(f"  Choose qubit LO at {qubitLO/1e9}GHz")
        # localconfig['qubitIF'] = fq - qubitLO
        # localconfig['qubitLO'] = qubitLO
        # localconfig['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency'] = qubitLO
        # localconfig['qmconfig']['elements']['qubit2']['mixInputs']['lo_frequency'] = qubitLO

        ##############
        # Qubit Spec 1D
        qubittuneconfig = deepcopy(localconfig)
        qubittuneconfig['qubit_output_gain'] = -10
        qubittuneconfig['saturation_amp'] = 10**(-20/20) * AMP0dBm
        qubittunefs = np.arange(-100e6, 100e6, 1e6) + max(np.round(localconfig['qubitIF']/10e6)*10e6, -350e6)
        qubittunefs = qubittunefs[qubittunefs >= -450e6]
        if len(qubittunefs) < 10:
            qubittunefs = np.arange(-450e6, -50e6, 2e6)

        prog = progs[11] = qmtools.QMQubitSpec(
            qmm, qubittuneconfig, Navg=1000,
            qubitIFs=qubittunefs)
        results['qubit:1'][i] = prog.run(plot=axs[0,4])
        try:
            fqIF = prog.find_dip(ax=axs[0,4])
            fq = fqIF + qubitLO
            print(f"  Qubit moved {(fqIF-localconfig['qubitIF'])/1e6:+.3f}MHz")
            localconfig['qubitIF'] = fqIF
            print(f"  Updated qubitIF to {localconfig['qubitIF']/1e6:.3f}MHz")
        except qmtools.PipelineException as e:
            print("Qubit update failed:", repr(e))
            print("Trying with stronger drive")
            qubittuneconfig['qubit_output_gain'] = 0
            qubittuneconfig['saturation_amp'] = 10**(-20/20) * AMP0dBm

            prog = progs[11] = qmtools.QMQubitSpec(
                qmm, qubittuneconfig, Navg=1000,
                qubitIFs=qubittunefs)
            results['qubit:1hp'][i] = prog.run(plot=axs[0,4])
            try:
                fqIF = prog.find_dip(ax=axs[0,4])
                fq = fqIF + qubitLO
                print(f"  Qubit moved {(fqIF-localconfig['qubitIF'])/1e6:+.3f}MHz")
                localconfig['qubitIF'] = fqIF
                print(f"  Updated qubitIF to {localconfig['qubitIF']/1e6:.3f}MHz")
            except qmtools.PipelineException as e:
                print("Qubit update failed, keep original estimate:", repr(e))

        #######
        # Readout SNR
        localconfig['saturation_amp'] = 0.316
        localconfig['qubit_output_gain'] = -15 #baseconfig['qubit_output_gain']
        localconfig['resonator_output_gain'] = 0 # baseconfig['resonator_output_gain'] # restore

        # prog = progs[4] = qmtools.QMReadoutSNR(
        #     qmm, localconfig, Navg=3e3,
        #     resonatorIFs=np.arange(199e6, 212e6, 0.1e6),
        #     readout_amps=np.logspace(np.log10(0.000316), np.log10(0.316), 19),
        #     drive_len=100)
        # results['readoutSNR'][i] = prog.run(plot=axs[1,2])

        localconfig['resonatorIF'] = int(bareIF)
        prog = progs[4] = qmtools.QMReadoutSNR_P1(
            qmm, localconfig, Navg=1e4,
            readout_amps=np.logspace(-60/20, 0, 37) * AMP0dBm,
            drive_len=100)
        results['readoutSNR_P1'][i] = res = prog.run(plot=axs[1,2])

        # Optimize readout power based on best SNR result
        try:
            bestamp, bestpower = prog.find_best_snr()
            if bestpower > -20:
                localconfig['resonator_output_gain'] = 0
            else:
                localconfig['resonator_output_gain'] = -20
            localconfig['short_readout_amp'] = 10**((bestpower - localconfig['resonator_output_gain'])/20) * AMP0dBm
        except qmtools.PipelineException as e:
            print("  ", repr(e))
            localconfig['resonator_output_gain'] = -13 # baseconfig['resonator_output_gain']
            localconfig['short_readout_amp'] = 0.1 # baseconfig['short_readout_amp']

        # restore qubit drive
        localconfig['saturation_amp'] = baseconfig['saturation_amp']
        localconfig['qubit_output_gain'] = baseconfig['qubit_output_gain']

        #######
        # Time Rabi
        #localconfig['cooldown_clk'] = 12500 # 50us
        #localconfig['cooldown_clk'] = 25000 # 100us

        localconfig['saturation_amp'] = 0.1
        localconfig['qubit_output_gain'] = -15 #baseconfig['qubit_output_gain']

        prog = progs[5] = qmtools.QMTimeRabi(
            qmm, localconfig, Navg=3e4,
            max_duration_ns=PROTOCOL_DURATION,
            drive_read_overlap_cycles=0)
        results['time_rabi:1'][i] = prog.run(plot=axs[0,3])

        localconfig['saturation_amp'] = 0.316
        localconfig['qubit_output_gain'] = -15 #baseconfig['qubit_output_gain']

        prog = progs[5] = qmtools.QMTimeRabi(
            qmm, localconfig, Navg=3e4,
            max_duration_ns=PROTOCOL_DURATION,
            drive_read_overlap_cycles=0)
        results['time_rabi:2'][i] = prog.run(plot=axs[0,3])

        localconfig['saturation_amp'] = 0.316
        localconfig['qubit_output_gain'] = 0

        prog = progs[5] = qmtools.QMTimeRabi(
            qmm, localconfig, Navg=3e4,
            max_duration_ns=PROTOCOL_DURATION,
            drive_read_overlap_cycles=0)
        results['time_rabi:3'][i] = prog.run(plot=axs[0,3])

        #######
        # Time Rabi Chevrons
        # (same plot as Time Rabi, need to decide or move somewhere else)
        localconfig['qubit_output_gain'] = -10
        localconfig['saturation_amp'] = 0.316

        axs[0,5].axhline(localconfig['qubitIF']/1e6, color='r', linestyle='--', linewidth=0.8, zorder=100)

        prog = progs[6] = qmtools.QMTimeRabiChevrons(
            qmm, localconfig, Navg=2e3,
            qubitIFs=np.arange(-400e6, 10e6, 10e6),
            max_duration_ns=PROTOCOL_DURATION,
            drive_read_overlap_cycles=0)
        results['time_rabi_chevrons:hpr'][i] = prog.run(plot=axs[0,5])

        # prog.clear_liveplot()
        # axs[0,5].axhline(localconfig['qubitIF']/1e6, color='r', linestyle='--', linewidth=0.8, zorder=100)

        # prog = progs[6] = qmtools.QMTimeRabiChevrons_InnerAvg(
        #     qmm, localconfig, Nrep=20, Navg=2e2,
        #     qubitIFs=np.arange(-300e6, 10e6, 10e6),
        #     max_duration_ns=PROTOCOL_DURATION,
        #     drive_read_overlap_cycles=0)
        # results['time_rabi_chevrons:inneravg'][i] = prog.run(plot=axs[0,5])

        # prog.clear_liveplot()

        # # Low power readout
        # localconfig['short_readout_amp'] = 0.056

        # prog = progs[6] = qmtools.QMTimeRabiChevrons(
        #     qmm, localconfig, Navg=1e5,
        #     qubitIFs=time_rabi_IFs,
        #     max_duration_ns=time_rabi_max_duration,
        #     drive_read_overlap_cycles=0)
        # results['time_rabi_chevrons'][i] = prog.run(plot=axs[0,3])

        # TODO fit chevrons to get good fq and drive power estimate

        ##############
        # Qubit Spec 1D
        prog = progs[11] = qmtools.QMQubitSpec(
            qmm, qubittuneconfig, Navg=500,
            qubitIFs=qubittunefs)
        results['qubit:2'][i] = prog.run(plot=axs[0,4])
        try:
            fqIF = prog.find_dip(ax=axs[0,4])
            fq = fqIF + qubitLO
            print(f"  Qubit moved {(fqIF-localconfig['qubitIF'])/1e6:+.3f}MHz")
            localconfig['qubitIF'] = fqIF
            print(f"  Updated qubitIF to {localconfig['qubitIF']/1e6:.3f}MHz")
        except qmtools.PipelineException as e:
            print("Qubit update failed, keep estimate:", repr(e))

        #######
        # Power Rabi, Gaussian
        # Note: max CW input to roomT amplifier is +4dBm
        localconfig['qubit_output_gain'] = 0 # max +4

        prog = progs[7] = qmtools.QMPowerRabi_Gaussian(
            qmm, localconfig, Navg=1e5, duration_ns=16, sigma_ns=4,
            drive_amps=np.linspace(0, AMP0dBm, 41))
        results['power_rabi:gaussian_8ns'][i] = prog.run(plot=axs[1,3])

        prog = progs[7] = qmtools.QMPowerRabi_Gaussian(
            qmm, localconfig, Navg=1e5, duration_ns=32, sigma_ns=8,
            drive_amps=np.linspace(0, AMP0dBm, 41))
        results['power_rabi:gaussian_16ns'][i] = prog.run(plot=axs[1,3])

        try:
            popt, perr = prog.fit_cosine(ax=axs[1,3], period0=0.13, plotp0=True)
            # Pi amp limited to +0dBm
            localconfig['pi_amp'] = min(10**(0/20)*AMP0dBm, popt[0] / 2)
        except Exception as e:
            print("Could not find pi pulse amplitude:", repr(e))
            continue

        #######
        # Ramsey Chevrons, Gaussian

        fc = np.round(localconfig['qubitIF']/10e6)*10e6
        ifs = np.arange(-110e6, 110e6, 5e6) + fc #np.linspace(-80e6, 80e6, 17) + fc
        prog = progs[9] = qmtools.QMRamseyChevronRepeat_Gaussian(
            qmm, localconfig, qubitIFs=ifs, Nrep=100, Navg=50,
            drive_len_ns=32, sigma_ns=8, readout_delay_ns=16,
            max_delay_ns=PROTOCOL_DURATION)
        results['ramsey_chevron_repeat:gaussian'][i] = prog.run(plot=axs[1,5])

        #######
        # Power Rabi, Square

        prog = progs[7] = qmtools.QMPowerRabi(
            qmm, localconfig, Navg=1e5, duration_ns=8,
            drive_amps=np.linspace(0, AMP0dBm, 41),
            drive_read_overlap_cycles=0)
        results['power_rabi:square_8ns'][i] = prog.run(plot=axs[1,3])

        prog = progs[7] = qmtools.QMPowerRabi(
            qmm, localconfig, Navg=1e5, duration_ns=16,
            drive_amps=np.linspace(0, AMP0dBm, 41),
            drive_read_overlap_cycles=0)
        results['power_rabi:square_16ns'][i] = prog.run(plot=axs[1,3])

        try:
            popt, perr = prog.fit_cosine(ax=axs[1,3], period0=0.13, plotp0=True)
            # Pi amp limited to +0dBm
            localconfig['pi_amp'] = min(10**(0/20)*AMP0dBm, popt[0] / 2)
        except Exception as e:
            print("Could not find pi pulse amplitude:", repr(e))
            continue
            # localconfig['pi_amp'] = 0.03

        #######
        # Ramsey
        # Uses pi pulse amplitude from config

        # prog = progs[9] = qmtools.QMRamsey(
        #     qmm, localconfig, Navg=5e4, drive_len_ns=8,
        #     max_delay_ns=60)
        # results['ramsey:on_resonance'][i] = prog.run(plot=axs[1,4])

        # # detune from qubit
        # localconfig['qubitIF'] = fq - qubitLO + 50e6

        # prog = progs[9] = qmtools.QMRamsey(
        #     qmm, localconfig, Navg=5e4, drive_len_ns=8,
        #     max_delay_ns=60)
        # results['ramsey:detuned'][i] = prog.run(plot=axs[1,5])

        # prog = progs[9] = qmtools.QMRamsey(
        #     qmm, localconfig, Navg=5e4, drive_len_ns=8,
        #     max_delay_ns=60)
        # results['ramsey:detuned_repetition'][i] = prog.run(plot=axs[1,4])

        fc = np.round(localconfig['qubitIF']/10e6)*10e6
        ifs = np.arange(-110e6, 110e6, 5e6) + fc #np.linspace(-80e6, 80e6, 17) + fc
        prog = progs[9] = qmtools.QMRamseyChevronRepeat(
            qmm, localconfig, qubitIFs=ifs, Nrep=50, Navg=100, drive_len_ns=16,
            max_delay_ns=PROTOCOL_DURATION)
        results['ramsey_chevron_repeat:square'][i] = prog.run(plot=axs[1,6])

        #######
        # Relaxation
        # Uses pi pulse amplitude from config

        # # retune onto qubit
        # localconfig['qubitIF'] = fq - qubitLO

        # prog = progs[8] = qmtools.QMRelaxation(
        #     qmm, localconfig, Navg=1e5, drive_len_ns=8,
        #     max_delay_ns=80)
        # results['relaxation'][i] = prog.run(plot=axs[0,4])

        # # High power readout
        # localconfig['short_readout_amp'] = 0.316

        prog = progs[8] = qmtools.QMRelaxation(
            qmm, localconfig, Navg=1e5, drive_len_ns=16,
            max_delay_ns=PROTOCOL_DURATION)
        results['relaxation:hpr'][i] = prog.run(plot=axs[1,4])

        estimator.step(i)
finally:
    estimator.end()
    print("Saving data")
    time.sleep(1) # Let user see the "saving" message to avoid KeyboardInterrupt

    if 'tof' not in globals():
        tof = None
    np.savez_compressed(
        fpath, Vgate=Vgate, results=results, baseconfig=baseconfig,
        resonator=resonator, resonatorfit=frfit, jpameta=jpameta,
        bareIF=bareIF, fq_estimate=fq_estimate,
        tof_precalibration=tof,
        measurement_duration=estimator.elapsed())
    fig.savefig(fpath+'.png')
    print("Data saved.")
