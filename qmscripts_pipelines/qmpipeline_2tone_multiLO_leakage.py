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

gate = BaselDACChannel(7)

assert gate.get_state(), "Basel channel not ON"
print("CH", gate.channel, ":", gate.get_voltage(), "V")

GATERAMP_STEP = 2e-6
GATERAMP_STEPTIME = 0.01

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
#%% Thermometry

from instruments.BlueforsPoll import BlueforsThermoPoll
from instruments.blueforsthermometer import BlueforsOldThermometer
# from instruments.blueforsthermometer import BlueforsNewThermometer

try:
    poll.stop()
except: pass

thermo = BlueforsOldThermometer()
# thermo = BlueforsNewThermometer()
poll = BlueforsThermoPoll.make_poll(thermo, ['MXC'], interval=10)

#%% Connect to oscilloscope for current measurement

import instruments.yokogawaoscillo
importlib.reload(instruments.yokogawaoscillo)
from instruments.yokogawaoscillo import YokogawaDL350
oscillo = YokogawaDL350("169.254.0.41")

# Test get voltage
oscillo.get_avg(1, 1)

TRANSAMP_GAIN = 1e9 # V/A = 1V/nA
CURRENT_LIMIT = 0.3e-9 # A

#%%

f2LOs = [2e9, 2.7e9, 3.4e9, 4.1e9, 4.8e9]

mixercal = qmtools.QMMixerCalibration(qmm, config_pipeline, qubitLOs=f2LOs)

#%%
importlib.reload(config_pipeline)
importlib.reload(qminit)

filename = '{datetime}_qmpipeline_2tone_LO_gateLeak'
fpath = data_path(filename, datesuffix='_qm')
Vgate = np.concatenate([np.linspace(0, -0.001, 11)])
Vstep = np.mean(np.abs(np.diff(Vgate)))
print(f"Vgate measurement step: {Vstep*1e6:.1f}uV avg")
Ngate = Vgate.size
assert len(Vgate) == 1 or Vstep > 1.19e-6, "Vstep smaller than Basel resolution"


fr_range = np.arange(200e6, 225e6, 0.05e6) # IF Hz

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

barefreq = 5.010e9 + 201.47e6#5.05578e9
bareIF = barefreq - baseconfig['resonatorLO']
GATE_SETTLETIME = 5 # s

# each name before colon indicates which program result is saved, i.e. the data format
# after the colon are identifiers if the same program is run multiple times (different settings)
results = {
    'mixer_calibration': [None]*Ngate,
    'resonator': [None]*Ngate,
    'resonator_P1': [None]*Ngate,
    'resonator_noise': [None]*Ngate,
    'qubit_P2': [None]*Ngate,
}
for lo in f2LOs:
    results[f'qubit_P2:LO{lo/1e9:.2f}GHz'] = [None]*Ngate
resonator = np.full((Vgate.size, fr_range.size), np.nan+0j)
frfit = np.full((Vgate.size, 4, 2), np.nan)

Vtransamp = np.full(Vgate.size, np.nan)
Ileak = np.full(Vgate.size, np.nan)
temp = {ch: np.full(Vgate.size, np.nan) for ch in poll.channels}

fig, axs = plt.subplots(nrows=3, ncols=2+int(np.ceil(len(f2LOs)/3)), layout='constrained', figsize=(16, 7))
frline, = axs[0,0].plot(Vgate, np.full(len(Vgate), np.nan), '.-', linewidth=1)
axs[0,0].set_xlabel('Vgate / V')
axs[0,0].set_ylabel('df / MHz')
axs[0,0].set_title('Cavity resonance vs gate')
#plots
lineIleak, = axs[2,0].plot(Vgate, Ileak, '.-')
axs[2,0].set_ylabel("Gate leakage / nA")
axs[2,0].set_xlabel("$V_g$ / V")
lineT, = axs[2,1].plot(Vgate, temp['MXC'], '.-')
axs[2,1].set_ylabel("MC Temperature / K")
axs[2,1].set_xlabel("$V_g$ / V")

progs = [None]*(4+len(f2LOs))
estimator = DurationEstimator(Ngate)
try:
    for i in range(Ngate):
        print(f"Setting gate to {Vgate[i]}V")
        gate.ramp_voltage(Vgate[i], GATERAMP_STEP, GATERAMP_STEPTIME)
        mpl_pause(GATE_SETTLETIME)
        
        oscillo.start()

        t = poll.get_all_channels()
        for ch in poll.channels:
            temp[ch][i] = t[ch][0]

        for j in range(len(progs)):
            if progs[j] is not None:
                progs[j].clear_liveplot()
                progs[j] = None

        results['mixer_calibration'][i] = mixercal.run_after_interval(3*3600)
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

        # #######
        # # Resonator noise
        # prog = progs[1] = qmtools.QMNoiseSpectrum(qmm, localconfig, Nsamples=100000, fcut_Hz=20e3)
        # results['resonator_noise'][i] = prog.run(plot=axs[1,1])
        # axs[1,1].set_xlim(-210,210)
        
        #######
        ## Leakage current
        Vtransamp[i] = oscillo.get_avg(1, 1)
        Ileak[i] = Vtransamp[i]/TRANSAMP_GAIN
        if abs(Ileak[i]) > CURRENT_LIMIT:
            print(f"Leakage detector over limit ({CURRENT_LIMIT*1e9}nA): {Vtransamp[i]}V = {Ileak[i]*1e9}nA")
            print("Ramping gate voltage to zero")
            gate.ramp_voltage(0, GATERAMP_STEP, GATERAMP_STEPTIME)
            break
        
        lineIleak.set_ydata(Ileak)
        axs[2,0].relim(), axs[2,0].autoscale(), axs[2,0].autoscale_view()
        lineT.set_ydata(temp['MXC'])
        axs[2,1].relim(), axs[2,1].autoscale(), axs[2,1].autoscale_view()

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
        # 2tone spectroscopy qubit vs power & update qubit IF
        # Note: have room-T amp, max input is +4dBm

        localconfig['qubit_output_gain'] = 0
    
        # for j, lo in enumerate(f2LOs):
        #     localconfig['qubitLO'] = lo
        #     # Need to set same LO for all elements on same mixer, otherwise calibration might not work
        #     localconfig['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency'] = lo
        #     localconfig['qmconfig']['elements']['qubit2']['mixInputs']['lo_frequency'] = lo
        #     # Note: Setting LO in qmconfig.mixers is without effect because that section
        #     # is overwritten with data from the calibration database by the QM manager.

        #     fq_amps = np.logspace(-60/20, -10/20, 6) * 0.3162
        #     assert np.max(fq_amps) <= 0.3161 # max 0dBm
        #     prog = progs[3+j] = qmtools.QMQubitSpec_P2(
        #         qmm, localconfig, Navg=100, qubitIFs=np.arange(-350e6, +350e6, 5e6),
        #         drive_amps=fq_amps)
        #     results[f'qubit_P2:LO{lo/1e9:.2f}GHz'][i] = prog.run(plot=axs[j%3,2+int(j/3)])

        estimator.step(i)
finally:
    estimator.end()
    print("Saving data")
    time.sleep(1) # Let uSavingser see the "saving" message to avoid KeyboardInterrupt

    if 'tof' not in globals():
        tof = None
    np.savez_compressed(
        fpath, Vgate=Vgate, Ileak=Ileak, T_MC=temp['MXC'], results=results, baseconfig=baseconfig,
        resonator=resonator, resonatorfit=frfit, jpameta=jpameta,
        tof_precalibration=tof, bareIF=bareIF,
        measurement_duration=estimator.elapsed())
    print("Data saved.")

    if Vgate.size >= 3:
        from vgatepipeline import VgatePipeline
        pl = VgatePipeline(fpath+'.npz')
        fig = pl.plot_Vgate_2tone_multi(figsize=(12, 8), npowers=5)
        fig.savefig(fpath+'.png')
