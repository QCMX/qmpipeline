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
jpapump = RsInstrument('TCPIP::169.254.2.32::INSTR', id_query=True, reset=False)


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
    2.0e9])
    # 2.0e9, 2.1e9, 2.2e9, 2.3e9, 2.4e9, 2.5e9, 2.6e9, 2.7e9, 2.8e9, 2.9e9,
    # 3.0e9, 3.1e9, 3.2e9, 3.3e9, 3.4e9, 3.5e9, 3.6e9, 3.7e9, 3.8e9, 3.9e9,
    # 4.0e9, 4.1e9, 4.2e9, 4.3e9, 4.4e9, 4.5e9, 4.6e9, 4.7e9, 4.8e9, 4.9e9,
    # 5.0e9, 5.1e9])

#%%
importlib.reload(config_pipeline)
importlib.reload(qminit)

filename = '{datetime}_qmpipeline_noise_only'
fpath = data_path(filename, datesuffix='_qm')
Vgate = np.array([-4.85])
#Vgate = np.linspace(-4.83, -4.85, 51)
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert len(Vgate) == 1 or Vstep > 1.19e-6, "Vstep smaller than Basel resolution"

fr_range = np.arange(199e6, 225e6, 0.05e6) # IF Hz

# Length / distance of pulses for Rabi / Ramsey / Relaxation
# It's nice to have the same length for everything to plot cohesive results later
PROTOCOL_DURATION = 400
T1DURATION = 800

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
    input("Info: JPA off [press enter]")

print(f"Setting gate ({abs(gate.get_voltage()-Vgate[0])/GATERAMP_STEP*GATERAMP_STEPTIME/60:.1f}min)")
gate.ramp_voltage(Vgate[0], GATERAMP_STEP, GATERAMP_STEPTIME)

baseconfig = qmtools.config_module_to_dict(config_pipeline)

# Mixer cal before other calibration
# TODO save result somewhere
initial_mixercal = mixercal.run_after_interval(6*3600)

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
    """Convert resonance shift (Hz, compared to n6.29o qubit) to qubit frequency (Hz)"""
    # Inverse of deltafr = g**2 * EC / (delta * (delta - EC)), delta = fq - fr
    EC = 0.35e9
    g = 0.23e9
    fr = 5.2e9
    return -np.sqrt(EC)*np.sqrt(EC*deltafr + 4*g**2)/(2*np.sqrt(deltafr)) + EC/2 + fr

GATE_SETTLETIME = 5 # s

# each name before colon indicates which program result is saved, i.e. the data format
# after the colon are identifiers if the same program is run multiple times (different settings)
results = {
    'mixer_calibration': [None]*Ngate,
    'resonator': [None]*Ngate,
    'resonator_P1': [None]*Ngate,
    'resonator_noise': [None]*Ngate,
}
resonator = np.full((Vgate.size, fr_range.size), np.nan+0j)
frfit = np.full((Vgate.size, 4, 2), np.nan)

fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained', figsize=(8, 5))
frline, = axs[0,0].plot(Vgate, np.full(len(Vgate), np.nan), '.-', linewidth=1)
axs[0,0].set_xlabel('Vgate / V')
axs[0,0].set_ylabel('df / MHz')
axs[0,0].set_title('Cavity resonance vs gate')

progs = [None]*2
estimator = DurationEstimator(Ngate)
try:
    for i in range(Ngate):
        estimator.step(i)
        print(f"Setting gate to {Vgate[i]}V")
        gate.ramp_voltage(Vgate[i], GATERAMP_STEP, GATERAMP_STEPTIME)
        mpl_pause(GATE_SETTLETIME)

        for j in range(len(progs)):
            if progs[j] is not None:
                progs[j].clear_liveplot()
                progs[j] = None

        # Don't run calibration. Might result in jumps in measured transmission & noise
        # mixercal.run_after_interval(6*3600)
        # results['mixer_calibration'][i] = True
        # Contains qm objects which we don't want to pickle
        #results['mixer_calibration'][i] = mixercal.run_after_interval(3*3600)

        localconfig = deepcopy(baseconfig)

        #######
        # One tone spectroscopy
        # localconfig['readout_amp'] = 0.0316 # -20dBm
        localconfig['qubit_output_gain'] = -15 # turn off qubit; -15dB is minimum for 2GHz

        prog = progs[0] = qmtools.QMResonatorSpec(qmm, localconfig, Navg=100, resonatorIFs=fr_range)
        results['resonator'][i] = resonatorspec = prog.run(plot=axs[1,0])
        resonator[i] = resonatorspec['Z']

        # Fit resonance, update resonator IF
        resonatorfit = prog.fit_lorentzian(ax=axs[1,0])
        frfit[i,:,0], frfit[i,:,1] = resonatorfit
        # show in Vgate plot
        frline.set_ydata((frfit[:,0,0] - bareIF)/1e6)
        axs[0,0].relim(), axs[0,0].autoscale(), axs[0,0].autoscale_view()

        if resonatorfit[0][1] < 100e3 or resonatorfit[0][1] > 5e6:
            print(f"  Resonator width {resonatorfit[0][1]/1e3}kHz out of good range. Skip.")
            mpl_pause(0.1)
            continue
        localconfig['resonatorIF'] = resonatorfit[0][0]

        #######
        # Resonator noise
        prog = progs[1] = qmtools.QMNoiseSpectrum(qmm, localconfig, Nsamples=1000000, fcut_Hz=None)
        results['resonator_noise'][i] = prog.run(plot=axs[1,1])
        axs[1,1].set_xlim(-210,210)

        res = results['resonator_noise'][0]
        fpoints = [50, 100]
        fftf = res['fftfreq']
        fidxs = [[np.argmin(np.abs(fftf-fp)), np.argmin(np.abs(fftf+fp))] for fp in fpoints]
        noise = np.full((i+1, len(fpoints)), np.nan)
        for j, res in enumerate(results['resonator_noise'][:i+1]):
            if res is None:
                continue
            absfft = np.abs(res['fft'])
            noise[j] = [np.sum(absfft[idxs[0]-1:idxs[0]+2] + absfft[idxs[1]-1:idxs[1]+2]) for idxs in fidxs]
        axs[0,1].clear()
        for j in range(len(fpoints)):
            axs[0,1].plot(Vgate[:i+1], noise[:,j], label=f'{fpoints[j]:.0f}Hz')
        axs[0,1].legend()
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

#%%

for r in results['ramsey_chevron_repeat:gaussian_16ns']:
    if r is not None:
        del r['config']['qmconfig']
