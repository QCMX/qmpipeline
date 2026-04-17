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
Vgate = np.linspace(-3.975, -3.985, 11)
# Vgate = np.array([-3.983])
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
    'readoutSNR_P1': [None]*Ngate,
    'qubit:1': [None]*Ngate,
    'qubit:1hp': [None]*Ngate,
    'qubit:2': [None]*Ngate,
    'time_rabi:1': [None]*Ngate,
    'time_rabi:2': [None]*Ngate,
    'time_rabi:3': [None]*Ngate,
    'time_rabi_chevrons': [None]*Ngate,
    'power_rabi:gaussian_8ns': [None]*Ngate,
    'power_rabi:gaussian_16ns': [None]*Ngate,
    'relaxation': [None]*Ngate,
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
        # estimate qubit LO, IF in range +100 to +200MHz
        deltafr = localconfig['resonatorIF'] - bareIF
        fqest = fq_estimate(deltafr)
        print(f"  Resonator shift: {deltafr/1e6}MHz")
        print(f"  Estimated qubit freq: {fqest/1e9}GHz")
        qubitLO = int(max(2e9, np.ceil((fqest-0.15e9)/1e8)*1e8))
        print(f"  Choose qubit LO at {qubitLO/1e9}GHz")
        localconfig['qubitIF'] = fqest - qubitLO
        localconfig['qubitLO'] = qubitLO
        localconfig['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency'] = qubitLO
        localconfig['qmconfig']['elements']['qubit2']['mixInputs']['lo_frequency'] = qubitLO

        #######
        # 2tone spectroscopy qubit vs power & update qubit IF
        # Note: have room-T amp, max input is +4dBm

        localconfig['qubit_output_gain'] = 0

        # Mark expected qubit freq in plot
        if fqest > 1.5e9:
            axs[0,2].axvline((fqest-qubitLO)/1e6, color='red', linestyle='--', linewidth=0.8, zorder=100)

        fq_amps = np.logspace(-50/20, 0, 11) * AMP0dBm
        assert np.max(fq_amps) <= 0.32 # max 0dBm
        prog = progs[3] = qmtools.QMQubitSpec_P2(
            qmm, localconfig, Navg=300, qubitIFs=np.arange(-450e6, +450e6, 2e6),
            drive_amps=fq_amps)
        results['qubit_P2'][i] = qubitspecvspower = prog.run(plot=axs[0,2])

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
