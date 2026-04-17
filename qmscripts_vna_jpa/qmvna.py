#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use QM as VNA over full frequency range.

Trick: To avoid rerunning a QUA program for each LO, we set the cailbration
parameters during a running program.

Scattering calibration
----------------------
The scattering parameter varies by more than 10dB from 20 to 400MHz even with
a flat cable, just due to the mixer properties. Therefore a calibration
can be loaded to correct the scattering parameter.

The calibration file has the structure

    [{'LO': float, 'IF': list of float, 'ReS': list, 'ImS': list}]

Mixer calibration
-----------------
The mixer calibration suppresses LO leakage and other side band.

The mixer calibration is loaded from the calibration file that is used and
created by the QuantumMachineManager. It is then applied during a running
sweep.

Note: With our OPX1 we cannot open multiple QM that share the same ports.
So we cannot open one QM per LO frequency.
"""
import os
import time
import json
import warnings
import numpy as np
import qm.qua as qua
import qm.octave as octave
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt

QMSLEEP = 0.01

class VNAException(Exception):
    pass


class QMVNA:
    def __init__(self, qmm, qmconfig: dict, element: str, outputgain: float,
                 los: list, ifs: list, Navg: int, phase_corr: float,
                 mixercalibrationdb: str = 'calibration_db.json',
                 Scalibration: str = 'vna_calibration.json'):
        self.qmconfig = qmconfig
        self.element = element
        self.qmm = qmm
        self.outputgain = outputgain
        self.los = np.array(los)
        self.ifs = np.array(ifs)
        self.Navg = Navg
        self.phase_corr = phase_corr
        self.mixercalibrationdbpath = mixercalibrationdb
        self.Scalibrationpath = Scalibration

        self.qm = None
        self.mixercalibration = None
        self.Scalibration = None

        self._make_programs()

        try:
            self.load_mixer_calibration()
        except VNAException as e:
            warnings.warn(f"Could not load mixer calibration from {self.mixercalibrationdbpath}: {repr(e)}")
        try:
            self.load_scattering_calibration()
        except VNAException as e:
            warnings.warn(f"Could not load scattering calibration from {self.Scalibrationpath}: {repr(e)}")

    def _make_programs(self):
        """Make qua programs and save as attributes."""
        # Sweep IF and average
        with qua.program() as self.sweep_prog:
            n = qua.declare(int)
            f = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.pause() # wait for mixer correction
            with qua.for_(n, 0, n < self.Navg, n + 1):
                with qua.for_(*from_array(f, self.ifs)):
                    qua.update_frequency(self.element, f)
                    # qua.wait(config.cooldown_clk, 'vna')  # wait for resonator to decay
                    qua.wait(rand.rand_int(50)+4, 'vna') # randomize demod error
                    qua.measure('readout', self.element, None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                    qua.save(I, I_st)
                    qua.save(Q, Q_st)

            with qua.stream_processing():
                I_st.buffer(len(self.ifs)).average().save('I')
                Q_st.buffer(len(self.ifs)).average().save('Q')

        # Play constant pulse until halted
        with qua.program() as self.prog_test:
            with qua.infinite_loop_():
                qua.play('const', self.element)

    def calibrate_mixers(self, calibrationif=150e6):
        """Run mixer calibration.

        Parameters
        ----------
        calibrationif : float, optional
            The default is 150e6.

        Returns
        -------
        Returns list of dicts matching `self.los`. Dict keys include
        `correction` for IQ correction matrix, `i_offset` and `q_offset`.
        """
        qm = self.qmm.open_qm(self.qmconfig)
        cal = []
        for lo in self.los:
            qm.octave.set_lo_frequency(self.element, lo)
            c = qm.octave.calibrate_element(self.element, [(lo, calibrationif)])
            c = list(c.values())[0].__dict__
            if c['correction'] == [1, 0, 0, 1]:
                warnings.warn(f"Calibration LO {lo/1e9:f}GHz IF{calibrationif/1e6:f}MHz failed (result is identity matrix).")
            cal.append(c)

        self.qm = None # Need to reopen
        self.mixercalibration = cal
        return cal

    def load_mixer_calibration(self, calibrationif=None, require_all_calibrated=True):
        """Load mixer calibration data from database.

        Uses last calibration value in database.
        Using `calibrationif` you can select calibrations at one specific IF.

        Will raise Exception if `require_all_calibrated=True` but not all
        `self.los` have calibration data available.

        Returns list of dicts matching `self.los`. Dict keys include
        `correction` for IQ correction matrix, `i_offset` and `q_offset`.
        """
        with open(self.mixercalibrationdbpath) as f:
            db = json.load(f)
        cal = [None]*len(self.los) # corresponding to self.los
        for i, lo in enumerate(self.los):
            for c in db['_default'].values():
                if c['lo_frequency'] == lo:
                    if calibrationif is not None:
                        if c['if_frequency'] == calibrationif:
                            cal[i] = c
                    else:
                        cal[i] = c
        for c in cal:
            if c is not None and c['correction'] == [1, 0, 0, 1]:
                warnings.warn(f"Calibration LO {c['lo_frequency']/1e9:f}GHz IF{c['if_frequency']/1e6:f}MHz failed (result is identity matrix).")
        if require_all_calibrated:
            if any(c is None for c in cal):
                raise VNAException("Mixer calibration DB does not contain all required values.")

        self.mixercalibration = cal
        return cal

    def test_mixer_cal(self, loidx):
        """Starts program to play constant pulse on output element.
        The interactively asks to apply first DC offsets then correction matrix.
        """
        self.qm = qm = self.qmm.open_qm(self.qmconfig)
        # Setup octave, output only
        qm.octave.set_lo_source(self.element, octave.OctaveLOSource.Internal)
        qm.octave.set_lo_frequency(self.element, self.los[loidx])
        qm.octave.set_rf_output_gain(self.element, self.outputgain)
        qm.octave.set_rf_output_mode(self.element, octave.RFOutputMode.on)
        # Start job
        job = qm.execute(self.prog_test)

        # Load calibration
        print(f"Job startd LO{self.los[loidx]/1e9:.2f}, calibration:")
        cal = self.mixercalibration
        print(cal[loidx])
        input("[enter] to set DC offsets")
        qm.set_output_dc_offset_by_element(self.element, 'I', cal[loidx]['i_offset'])
        qm.set_output_dc_offset_by_element(self.element, 'Q', cal[loidx]['q_offset'])
        input("[enter] to set mixer correction matrix")
        job.set_element_correction(self.element, cal[loidx]['correction'])
        input("Calibration applied. [enter] halt")
        job.halt()
        return qm, job

    def load_scattering_calibration(self):
        if not os.path.exists(self.Scalibrationpath):
            raise VNAException("Scattering calibration file not found.")
        with open(self.Scalibrationpath) as f:
            calfile = json.load(f)
        cal = [None]*len(self.los)
        for i, lo in enumerate(self.los):
            for rec in calfile:
                if rec['LO'] == lo:
                    mask = np.any(np.array(rec['IF'])[:,None] == self.ifs[None,:], axis=1)
                    if np.count_nonzero(mask) == self.ifs.size:
                        cal[i] = np.array(rec['ReS'])[mask] + 1j * np.array(rec['ImS'])[mask]
        if any(c is None for c in cal):
            raise VNAException("Scattering calibration DB does not contain all required values.")
        self.Scalibration = np.array(cal)
        return cal

    def calibrate_scattering(self, plot=True, write_to_file=True):
        """
        Run sweep without corrections and save result into calibration file.
        Overwrites existing values in calibration.

        The setup should be changed to a calibrated connection between RF out
        and RF in.
        """
        S = self.sweep(apply_calibration=False)
        cal = [{'LO': self.los[i], 'IF': list(self.ifs),
                'ReS': list(S[i].real), 'ImS': list(S[i].imag)}
               for i  in range(S.shape[0])]
        self.Scalibration = cal
        if write_to_file:
            with open(self.Scalibrationpath, "w+") as fp:
                json.dump(cal, fp)
        if plot:
            self.plot(S)
        return cal

    def _prepare_sweep_job(self):
        """Make QUA program to sweep IFs and compile on QM.

        Returns cached QM and precompiled job if available.
        Opening other quantum machines between calling the VNA means that
        a new QM needs to be opened and the job to be compiled again.
        """
        if self.qm is not None:
            if self.qm.id in self.qmm.list_open_quantum_machines():
                return self.qm, self.prog_id

        self.qm = qm = self.qmm.open_qm(self.qmconfig)
        self.prog_id = prog_id = qm.compile(self.sweep_prog)
        return qm, prog_id

    def sweep(self, output_off_after=True, apply_calibration=True):
        """Sweep IF with given values at each LO.

        Returns
        -------
        2D np.ndarray of S, first axis LO, second axis IF.
        Phase corrected and amplitude in units of volts.
        """
        if apply_calibration and self.Scalibration is None:
            raise VNAException("No S calibration loaded.")

        qm, prog_id = self._prepare_sweep_job()

        # Octave upconversion
        qm.octave.set_lo_source(self.element, octave.OctaveLOSource.Internal)
        qm.octave.set_lo_frequency(self.element, self.los[0])
        qm.octave.set_rf_output_gain(self.element, self.outputgain)
        qm.octave.set_rf_output_mode(self.element, octave.RFOutputMode.on)
        # Octave downconversion
        qm.octave.set_qua_element_octave_rf_in_port(self.element, "octave1", 1)
        qm.octave.set_downconversion(
            self.element, lo_source=octave.RFInputLOSource.Internal, lo_frequency=self.los[0],
            if_mode_i=octave.IFMode.direct, if_mode_q=octave.IFMode.direct
        )

        S = np.full((len(self.los), len(self.ifs)), np.nan+0j)
        cal = self.mixercalibration
        # mixername = self.qmconfig['elements'][self.element]['mixInputs']['mixer']
        for i, lo in enumerate(self.los):
            job = qm.queue.add_compiled(self.prog_id).wait_for_execution()
            while not job.is_paused():
                time.sleep(QMSLEEP)
            # Set calibration
            qm.octave.set_lo_frequency(self.element, self.los[i])
            qm.set_output_dc_offset_by_element(self.element, 'I', cal[i]['i_offset'])
            qm.set_output_dc_offset_by_element(self.element, 'Q', cal[i]['q_offset'])
            qm.octave.set_downconversion(
                self.element, lo_source=octave.RFInputLOSource.Internal, lo_frequency=self.los[i],
                if_mode_i=octave.IFMode.direct, if_mode_q=octave.IFMode.direct
            )
            #qm.set_mixer_correction(mixername, self.los[i], calibrationif)
            job.set_element_correction(self.element, cal[i]['correction'])
            # Run
            job.resume()
            job.result_handles.wait_for_all_values()
            # Get data
            I = job.result_handles.get('I').fetch_all()
            Q = job.result_handles.get('Q').fetch_all()
            f = lo + self.ifs
            readoutlen = self.qmconfig['pulses']['readout_pulse']['length']
            S[i] = (I + 1j*Q) * 2**12 / readoutlen * np.exp(1j * f * self.phase_corr)

        if output_off_after:
            qm.octave.set_rf_output_mode(self.element, octave.RFOutputMode.off)
        if apply_calibration:
            S = S / self.Scalibration
        return S

    def octave_out_off(self):
        if self.qm is None:
            qm = self.qmm.open_qm(self.qmconfig)
        else:
            qm = self.qm
        qm.octave.set_rf_output_mode(self.element, octave.RFOutputMode.off)

    def plot(self, S):
        fig, axs = plt.subplots(nrows=2, sharex=True, layout='constrained')
        argS = np.unwrap(np.angle(S).reshape(-1)).reshape(S.shape)
        for i, lo in enumerate(self.los):
            f = lo + self.ifs
            t = f"LO {lo/1e9:.2f} GHz"
            axs[0].plot(f/1e9, 10*np.log10(np.abs(S[i])**2 * 10), label=t)
            axs[1].plot(f/1e9, argS[i], label=t)
        axs[0].set_ylabel("|S| / dBm at ADC in")
        axs[1].set_ylabel("arg S")
        axs[1].set_xlabel("Frequency / GHz")
        for ax in axs.flat:
            #ax.legend(fontsize=6)
            ax.grid()

        readoutlen = self.qmconfig['pulses']['readout_pulse']['length']
        readoutamp = self.qmconfig['waveforms']['readout_wf']['sample']
        readoutpower = 10*np.log10(readoutamp**2 * 10) # V to dBm
        inputgain = self.qmconfig['controllers']['con1']['analog_inputs'][1]['gain_db'] # dB
        fig.suptitle(
            f"   Navg {self.Navg}"
            f"   electric delay {self.phase_corr:.3e}rad/Hz"
            f"\n{readoutlen/1e3}us readout at {readoutpower:.1f}dBm{self.outputgain:+.1f}dB"
            f",   {inputgain:+.1f}dB input gain",
            fontsize=10)
        return fig, axs


if __name__ == '__main__':
    import importlib

    import configuration_vna as config
    importlib.reload(config)

    import qminit
    qmm = qminit.connect()

    los = np.arange(2e9, 10.1e9, 300e6)# [4.5e9, 5e9, 5.5e9, 6e9, 6.5e9, 7e9, 7.5e9]
    ifs = np.arange(10e6, 451e6, 1e6)
    vna = QMVNA(qmm, config.qmconfig, 'vna', outputgain=0, los=los, ifs=ifs, Navg=100, phase_corr=1.6239384827049928e-06)
    #S = vna.sweep()
    #vna.plot(S)
