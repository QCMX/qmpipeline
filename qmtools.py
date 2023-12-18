#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scriptable set of quantum machine programs for calibration and measurement pipelines.

Need to propagate KeyboardInterrupt to stop pipeline.
But significant data processing might be done in finally-statement.

Data saving is part of the higher level structure, not here, because there
might be additional meta-data to go with each program's context.

Configuration
=============
Not independent of configuration! E.g. names of waveforms are hard coded.

Need configuration that
- contains quantum machine config and waveform information
- contains octave settings
- may be modified to adjust frequencies, amplitudes and possibly waveform lengths
- may be modified by baking new waveforms
- saved alongside data.
Therefore
- needs to be python primitive type that can be deep-copied to allow modifications
- mostly non-redundant to avoid inconsistencies after update waveform properties

If qmconfig is unchanged for all programs, we could reuse the same QuantumMachine object for all programs.
This is possible for all programs that only change
- pulse length in 4ns resolution
- pulse amplitude
- intermediate frequency, inside calibrated regions
by modifying pulses in the qua program. Overwritten at the top-level of the config file
while the original values are in config.qmconfig.

It is not obvious which parameters should be subject to change on program-bases
and which are fixed only in the configuration because it depends on the
purpose of the pipeline.

The number of averages can be handled as instance variable and the
actual number of averages (different if program is interrupted) is saved in
the result set.

Waveforms
=========
Which waveforms are used by which program. Readout:
- readout (10us): time of flight, resonator spec (vs readout power), qubit spec (vs drive power)
- short_readout (100ns): time Rabi (vs f2 / P2), power Rabi, relaxation, Ramsey
Drive:
- saturation (10us): qubit spec, time Rabi
- pi (rectangle): 
- arbitrary, generated with program parameters: time Rabi, power Rabi


Parameters
==========
Time of flight: readout amplitude, gain, resonator IF (not readout length)
Resonator spec: readout amplitude, gain, RANGE: resonator IF (not readout length)
Qubit spec: readout amplitude, gain, resonator IF, qubit LO, qubit IF, saturation amp, saturation gain (not readout / saturation length)
Time Rabi: readout amplitude, gain, resonator IF, qubit LO, qubit IF, drive amp, drive gain, RANGE: drive len
Power Rabi: ...
"""

import time
import inspect
import warnings
from copy import deepcopy
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import qm.qua as qua
import qm.octave as octave
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.bakery import baking

import qminit
from helpers import mpl_pause


def config_module_to_dict(module):
    return {item: getattr(module, item) for item in dir(module) if not (
        item.startswith("__") or item == 'meta'
        or inspect.ismodule(getattr(module, item))  # exclude modules
        or hasattr(getattr(module, item), '__call__')  # exclude functions
    )}

def opx_amp2pow(amp, ext_gain_db=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return 10*np.log10(amp**2 * 10) + ext_gain_db
def opx_pow2amp(pow, ext_gain_db=0):
    return np.sqrt(10**((pow-ext_gain_db)/10) / 10)


class PipelineException (Exception):
    pass

class ResultUnavailableError (PipelineException):
    pass


class QMProgram (object):
    """One program that can be run with some parameters.

    Programs are initialized with parameters like timing, ranges, etc.
    And they are treated as immutable after instance creation.
    I.e. the result of get_meta() will not change.
    """

    def __init__(self, qmm, config):
        self.qmm = qmm
        if inspect.ismodule(config):
            self.baseconfig = config_module_to_dict(config)
        else:
            self.baseconfig = config
        self.config = deepcopy(self.baseconfig)
        self.params = {}

    def get_meta(self):
        return self.config

    def _make_program(self):
        """Needs to be implemented by subclass.

        Defines self.qmprog for compilation or execution later.
        """
        raise NotImplementedError

    def compile(self):
        if hasattr(self, 'progid'):
            return self.progid
        if not hasattr(self, 'qmprog'):
            self._make_program()
        qm = self.qmm.open(self.config['qmconfig'])
        self.progid = qm.compile(self.qmprog)
        return self.progid

    def run(self, plot=None, printn=True, pause=1):
        """Run program and return result data.

        Will automatically print iteration numbers and time estimate
        if 'iteration' handle is present, unless 'printn' is False.

        'plot' may be True for a live plot in a dedicated window or
        may be an axes object that will be used for plotting.

        'pause' is the sleep time (in seconds) between checking the
        program results and updating the live plot.
        """
        # Note: if baking, _make_program may change qmconfig
        # So need to do this before opening QM
        if not hasattr(self, 'qmprog'):
            self._make_program()

        # TODO: reuse open quantum machine if config is the same?
        qm = self.qm = self.qmm.open_qm(self.config['qmconfig'])
        self._init_octave(qm)
        # TODO: precompile, but then needs to run on same qm
        # progid = self.compile()
        # pendingjob = qm.queue.add_compiled(progid)
        # job = pendingjob.wait_for_execution()

        job = self.last_job = qm.execute(self.qmprog)
        resulthandles = job.result_handles
        tstart = time.time()

        if isinstance(plot, bool) and plot == True:
            fig, ax = plt.subplots(layout='constrained')
        elif isinstance(plot, mpl.axes.Axes):
            ax = plot
        else:
            ax = None
        if ax:
            self._initialize_liveplot(ax)
            if isinstance(plot, bool):
                fig.show()

        hasiter = 'iteration' in resulthandles._all_results
        Navg = self.params['Navg'] if 'Navg' in self.params else np.nan
        try:
            while resulthandles.is_processing():
                if hasiter:
                    iteration = resulthandles.iteration.fetch_all() or 1
                    print(
                        f"iteration={iteration}, remaining: {(Navg-iteration) * (time.time()-tstart)/iteration:.0f}s")
                if ax:
                    self._update_liveplot(ax, resulthandles)
                mpl_pause(pause)
        except Exception as e:
            print("Halting QuantumMachine job due to Exception")
            job.halt()
            raise e
        except KeyboardInterrupt:
            print("Halting QuantumMachine job due to user interrupt")
            job.halt() # continue processing results

        trun = time.time() - tstart
        resulthandles.wait_for_all_values()
        print(f"Execution time: {trun:.1f}s, exit status: {job.status}")
        # if job.status != 'completed':
        print("Job execution report:", job.execution_report())

        if ax:
            self._update_liveplot(ax, resulthandles)
        result = self._retrieve_results(resulthandles)
        return result

    def _init_octave(self, qm):
        """Initializes octave LO and gain given settings in self.params
        (instance specific) or self.config as fallback."""
        element = 'resonator'
        lofreq = self.params.get('resonatorLO', self.config['resonatorLO'])
        qm.octave.set_lo_source(element, octave.OctaveLOSource.Internal)
        qm.octave.set_lo_frequency(element, lofreq)
        qm.octave.set_rf_output_gain(element, self.params.get(
            'resonator_output_gain', self.config['resonator_output_gain']))
        qm.octave.set_rf_output_mode(element, octave.RFOutputMode.on)
        qm.octave.set_qua_element_octave_rf_in_port(element, "octave1", 1)
        qm.octave.set_downconversion(
            element, lo_source=octave.RFInputLOSource.Internal, lo_frequency=lofreq,
            if_mode_i=octave.IFMode.direct, if_mode_q=octave.IFMode.direct
        )

        element = 'qubit'
        lofreq = self.params.get('qubitLO', self.config['qubitLO'])
        qm.octave.set_lo_source(element, octave.OctaveLOSource.Internal)
        # qm.octave.set_lo_source(element, octave.OctaveLOSource.LO2)
        qm.octave.set_lo_frequency(element, lofreq)
        qm.octave.set_rf_output_gain(element, self.params.get(
            'qubit_output_gain', self.config['qubit_output_gain']))
        qm.octave.set_rf_output_mode(element, octave.RFOutputMode.on)

    def _retrieve_results(self, resulthandles=None):
        """Dictionary with values of all output streams and input parameters.
        Magically combines I and Q into complex Z.

        Values of results may be None if the streams have not yet produced
        any values. A ResultUnavailableError is raised when no job has been
        run previously.
        """
        if resulthandles is None:
            if hasattr(self, 'last_job'):
                resulthandles = self.last_job.result_handles
            else:
                raise ResultUnavailableError("No results cached yet.")
        res = {k: (f.fetch_all() if len(f) else None) for k, f in resulthandles._all_results.items()}
        # Copy results because they are mutable and we don't want to contaminate
        # the original results in job.result_handles
        res = {k: (np.copy(a) if isinstance(a, np.ndarray) else a)
               for k, a in res.items()}
        # if any(v is None for v in res.values()):
        #     return None
        if 'I' in res and 'Q' in res and 'Z' not in res:
            if res['I'] is not None and res['Q'] is not None:
                res['Z'] = (res['I'] + 1j * res['Q'])
            else:
                res['Z'] = None
        return self.params | res | {'config': self.config}

    def _initialize_liveplot(self, ax):
        pass

    def _update_liveplot(self, ax, resulthandles):
        pass
    
    def clear_liveplot(self):
        if hasattr(self, 'colorbar'):
            try:
                self.colorbar.remove()
            except:
                pass
        if hasattr(self, 'ax'):
            self.ax.clear()


class QMMixerCalibration (QMProgram):
    """
    Runs calibration at LO and IF given in the config.
    For the qubit runs calibration at additional LO frequencies
    supplied as parameter (with the IF from the config).

    By default the calibration database should be active in the
    QuantumMachineManager so the latest calibration is automatically applied.

    In the QM config, the LO and IF for the element and in the mixer definition have to match,
    otherwise an error is produced when opening the quantum machine.
    The QM selects the calibration matching the LO and IF given in the config
    upon opening the quantum machine.
    """

    def __init__(self, qmm, config, qubitLOs=None):
        super().__init__(qmm, config)
        if qubitLOs is None:
            qubitLOs = [self.config['qubitLO']]
        self.params = {'qubitLOs': qubitLOs}

    def _make_program(self):
        pass

    def compile(self):
        raise NotImplementedError

    def run(self):
        qm = self.qmm.open_qm(self.config['qmconfig'])
        self._init_octave(qm)
        print("Running calibration on resonator channel...")
        rcal = qm.octave.calibrate_element(
            'resonator', [(self.config['resonatorLO'], self.config['resonatorIF'])])
        qubitLOs = list(self.params['qubitLOs'])
        if self.config['qubitLO'] not in qubitLOs:
            qubitLOs.append(self.config['qubitLO'])
        print(f"Running calibration on qubit channel for {len(self.params['qubitLOs'])} LO frequencies...")
        qcal = []
        for lof in qubitLOs:
            qm.octave.set_lo_frequency('qubit', lof)
            qcal.append(qm.octave.calibrate_element(
                'qubit', [(lof, self.config['qubitIF'])]))
        return {'resonator': rcal, 'qubit': qcal} | self.params | {'config': self.config}


class QMTimeOfFlight (QMProgram):
    def __init__(self, qmm, config, Navg):
        super().__init__(qmm, config)
        self.Navg = Navg
        self.params = {'Navg': Navg}

    def _make_program(self):
        amp_scale = self.config['readout_amp'] / \
            self.config['qmconfig']['waveforms']['readout_wf']['sample']
        assert self.config['readout_len'] == self.config['qmconfig']['pulses']['readout_pulse']['length']

        with qua.program() as prog:
            n = qua.declare(int)
            n_st = qua.declare_stream()
            adc_st = qua.declare_stream(adc_trace=True)

            qua.update_frequency('resonator', self.config['resonatorIF'])

            with qua.for_(n, 0, n < self.Navg, n + 1):
                qua.reset_phase('resonator')
                # qua.play('preload', 'resonator')
                qua.measure('readout'*qua.amp(amp_scale), 'resonator', adc_st)
                qua.wait(self.config['cooldown_clk'], 'resonator')
                qua.save(n, n_st)

            with qua.stream_processing():
                n_st.save('iteration')
                adc_st.input1().average().save('adc1')
                adc_st.input2().average().save('adc2')
                # Will save only last run:
                adc_st.input1().save('adc1_single_run')
                adc_st.input2().save('adc2_single_run')
        self.qmprog = prog
        return prog

    def apply_offset_to(self, config, printinfo=True):
        tof = self._retrieve_results()
        nsamples = tof['adc1'].size
        offserr = np.mean(tof['adc1']), np.mean(tof['adc2'])
        newoffs1 = config['qmconfig']['controllers']['con1']['analog_inputs'][1]['offset'] - np.mean(tof['adc1'])/2**12
        newoffs2 = config['qmconfig']['controllers']['con1']['analog_inputs'][2]['offset'] - np.mean(tof['adc2'])/2**12
        if printinfo:
            print('    Single run noise:', np.std(tof['adc1_single_run']), np.std(tof['adc1_single_run']), 'ADC units')
            print('    Exp. avg. noise:', np.std(tof['adc1_single_run'])/np.sqrt(tof['Navg']), np.std(tof['adc2_single_run'])/np.sqrt(tof['Navg']), 'ADC units')
            #print('Averaged noise:', np.std(np.diff(tof['adc1']))/np.sqrt(2), np.std(np.diff(tof['adc2']))/np.sqrt(2))
            print('    Averaged noise:', np.std(tof['adc1']), np.std(tof['adc1']), 'ADC units')
            print('    Offset error:', offserr, 'ADC')
            print('    Offset error uncertainty:', np.std(tof['adc1'])/np.sqrt(nsamples), np.std(tof['adc2'])/np.sqrt(nsamples), 'ADC samples')
            print('    Offset correct to:', newoffs1, newoffs2, 'V')
        config['qmconfig']['controllers']['con1']['analog_inputs'][1]['offset'] = newoffs1
        config['qmconfig']['controllers']['con1']['analog_inputs'][2]['offset'] = newoffs2


class QMResonatorSpec (QMProgram):
    def __init__(self, qmm, config, Navg, resonatorIFs):
        super().__init__(qmm, config)
        self.params = {'resonatorIFs': resonatorIFs, 'Navg': Navg}

    def _make_program(self):
        amp_scale = self.config['readout_amp'] / \
            self.config['qmconfig']['waveforms']['readout_wf']['sample']
        assert self.config['readout_len'] == self.config['qmconfig']['pulses']['readout_pulse']['length']
        freqs = self.params['resonatorIFs']

        with qua.program() as prog:
            n = qua.declare(int)
            f = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            n_st = qua.declare_stream()
            rand = qua.lib.Random()

            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                with qua.for_(*from_array(f, freqs)):
                    qua.update_frequency('resonator', f)
                    qua.wait(self.config['cooldown_clk'], 'resonator')
                    # randomize demod error
                    qua.wait(rand.rand_int(50)+4, 'resonator')
                    qua.measure('readout'*qua.amp(amp_scale), 'resonator', None,
                                qua.dual_demod.full(
                                    'cos', 'out1', 'sin', 'out2', I),
                                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                    qua.save(I, I_st)
                    qua.save(Q, Q_st)
                qua.save(n, n_st)

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(len(freqs)).average().save('I')
                Q_st.buffer(len(freqs)).average().save('Q')

        self.qmprog = prog
        return prog

    def _retrieve_results(self, resulthandles=None):
        """Applies phase correction to Z."""
        res = super()._retrieve_results(resulthandles)
        nsamples = self.config['qmconfig']['pulses']['readout_pulse']['length']
        # Note: don't use (a *= 3) because it modifies the result in the iterator.
        if res['I'] is not None:
            res['I'] = res['I'] * (2**12 / nsamples)
        if res['Q'] is not None:
            res['Q'] = res['Q'] * (2**12 / nsamples)
        if res['Z'] is not None:
            res['Z'] = res['Z'] * (2**12 / nsamples)
            res['Z'] *= np.exp(1j * self.params['resonatorIFs'] * self.config['PHASE_CORR'])
        return res

    def _initialize_liveplot(self, ax):
        freqs = self.params['resonatorIFs']
        line, = ax.plot(freqs/1e6, np.full(len(freqs), np.nan), label="|S|")
        readoutpower = opx_amp2pow(self.config['readout_amp'])
        ax.set_title(
            "resonator spectroscopy analysis\n"
            f"LO {(self.config['resonatorLO'])/1e9:f}GHz"
            f"   Navg {self.params['Navg']}\n"
            f"{self.config['readout_len']/1e3:.0f}us readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB",
            fontsize=8)
        ax.set_xlabel('IF  [MHz]')
        ax.set_ylabel('|S|  [Volt]')
        self.line = line
        self.ax = ax # for self.clear_liveplot

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.abs(res['Z']))
        ax.relim(), ax.autoscale(), ax.autoscale_view()

    @staticmethod
    def lorentzian_amplitude(f, f0, width, a, tau0):
        """Complex 'squareroot' of the Lorentzian."""
        tau = 0
        L = (width/2) / ((width/2) + 1j*(f - f0))
        return (a * np.exp(1j*(tau0 + tau*(f-f0))) * L).view(float)

    def fit_lorentzian(
            self, result=None, ax=None, plotp0=False,
            pltkw={'color': 'k', 'linewidth': 1}, printinfo=True):
        """Fit 'square root' lorentzian shape to data.

        results is the return value of run() or None to use the results of the last run().

        Returns tuple of (optimal, uncertainty), each being an array of
        f0, width, amplitude and angle offset.
        """
        res = self._retrieve_results() if result is None else result
        freqs = res['resonatorIFs']
        func = QMResonatorSpec.lorentzian_amplitude
        Z = res['Z']
        p0 = self.last_p0 = [
            (np.mean(freqs[:5]) + np.mean(freqs[-5:])) / 2,  # f0
            (np.max(freqs[-1])-np.min(freqs[0])) / 3,  # width
            np.max(np.abs(Z)),  # amplitude
            np.angle(Z[np.argmax(np.abs(Z))])  # angle
        ]
        popt, pcov = curve_fit(
            func, freqs, Z.view(float), p0=p0,
            bounds=(
                [np.min(freqs), np.diff(freqs)[0], 0, -np.inf],
                [np.max(freqs), np.max(freqs)-np.min(freqs), np.inf, np.inf]))
        perr = np.sqrt(np.diag(pcov))
        if printinfo:
            res = [ufloat(opt, err)
                   for opt, err in zip(popt, np.sqrt(np.diag(pcov)))]
            for r, name in zip(res, ["f0", "width", "a", "tau"]):
                print(f"    {name:6s} {r}")
        if ax:
            ax.plot(freqs/1e6, np.abs(func(freqs, *popt).view(complex)), '-', **pltkw)
            if plotp0:
                ax.plot(freqs/1e6, np.abs(func(freqs, *p0).view(complex)), '--', **pltkw)
        return popt, perr


class QMResonatorSpec_P2 (QMProgram):
    def __init__(self, qmm, config, Navg, resonatorIFs, readout_amps):
        super().__init__(qmm, config)
        self.params = {
            'resonatorIFs': resonatorIFs,
            'readout_amps': readout_amps,
            'Navg': Navg
        }

    def _make_program(self):
        amps = self.params['readout_amps'] / \
            self.config['qmconfig']['waveforms']['readout_wf']['sample']
        assert np.all(amps > 0), "Only positive amplitudes are valid."
        assert np.all(amps <= 2-2e-16), "Exceeding maximum value of amplitude scaling."
        assert self.config['readout_len'] == self.config['qmconfig']['pulses']['readout_pulse']['length']
        freqs = self.params['resonatorIFs']

        with qua.program() as prog:
            n = qua.declare(int)
            f = qua.declare(int)
            a = qua.declare(qua.fixed)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            n_st = qua.declare_stream()
            rand = qua.lib.Random()

            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                with qua.for_(*from_array(f, freqs)):
                    qua.update_frequency('resonator', f)
                    with qua.for_(*from_array(a, amps)):
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator') # randomize demod error
                        qua.measure('readout'*qua.amp(a), 'resonator', None,
                                    qua.dual_demod.full(
                                        'cos', 'out1', 'sin', 'out2', I),
                                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                qua.save(n, n_st)

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(len(freqs), len(amps)).average().save('I')
                Q_st.buffer(len(freqs), len(amps)).average().save('Q')

        self.qmprog = prog
        return prog

    def _retrieve_results(self, resulthandles=None):
        """Applies phase correction to Z."""
        res = super()._retrieve_results(resulthandles)
        nsamples = self.config['qmconfig']['pulses']['readout_pulse']['length']
        # Note: don't use (a *= 3) because it modifies the result in the iterator.
        if res['I'] is not None:
            res['I'] = res['I'] * (2**12 / nsamples)
        if res['Q'] is not None:
            res['Q'] = res['Q'] * (2**12 / nsamples)
        if res['Z'] is not None:
            res['Z'] = res['Z'] * (2**12 / nsamples)
            res['Z'] *= np.exp(1j * self.params['resonatorIFs'][:,None] * self.config['PHASE_CORR'])
        return res

    def _initialize_liveplot(self, ax):
        freqs = self.params['resonatorIFs']
        amps = self.params['readout_amps']  # V
        power = opx_amp2pow(amps, self.config['resonator_output_gain'])

        xx, yy = np.meshgrid(freqs/1e6, power, indexing='ij')
        self.img = ax.pcolormesh(xx, yy, np.full(
            (len(freqs), len(amps)), np.nan), shading='nearest')
        ax.set_title(
            "resonator spectroscopy vs readout power\n"
            f"LO {(self.config['resonatorLO'])/1e9:f}GHz"
            f"   Navg {self.params['Navg']}\n"
            f"{self.config['readout_len']/1e3:.0f}us readout,"
            f"  {self.config['resonator_output_gain']:+.1f}dB output gain",
            fontsize=8)
        ax.set_xlabel("readout IF / MHz")
        ax.set_ylabel("readout power / dBm")
        # ax.get_figure().colorbar(self.img, ax=ax).set_label('amplitude / V')

        axright = ax.secondary_yaxis(
            'right', functions=(
                lambda p: opx_pow2amp(p, self.config['resonator_output_gain']),
                lambda a: opx_amp2pow(a, self.config['resonator_output_gain'])))
        axright.set_ylabel('readout amplitude / V')

        self.ax = ax
        self.axright = axright

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        amps = self.params['readout_amps']
        # Note: setting an empty Normalize resets the colorscale
        self.img.set(array=np.abs(res['Z'])/amps[None,:], norm=mpl.colors.Normalize())

    def clear_liveplot(self):
        if hasattr(self, 'ax'):
            self.ax.clear()


class QMQubitSpec (QMProgram):
    def __init__(self, qmm, config, Navg, qubitIFs):
        super().__init__(qmm, config)
        self.params = {'qubitIFs': qubitIFs, 'Navg': Navg}

    def _make_program(self):
        read_amp_scale = self.config['readout_amp'] / \
            self.config['qmconfig']['waveforms']['readout_wf']['sample']
        drive_amp_scale = self.config['saturation_amp'] / \
            self.config['qmconfig']['waveforms']['saturation_wf']['sample']
        assert self.config['saturation_len'] == self.config['qmconfig']['pulses']['saturation_pulse']['length']
        assert self.config['readout_len'] == self.config['qmconfig']['pulses']['readout_pulse']['length']
        freqs = self.params['qubitIFs']

        assert self.config['saturation_len'] >= self.config['readout_len']
        readoutwait = int(
            (self.config['saturation_len'] - self.config['readout_len']) / 2 / 4)  # cycles

        with qua.program() as prog:
            n = qua.declare(int)
            f = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            n_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                with qua.for_(*from_array(f, freqs)):
                    qua.update_frequency('qubit', f)
                    qua.wait(self.config['cooldown_clk'], 'resonator')
                    # randomize demod error
                    qua.wait(rand.rand_int(50)+4, 'resonator')
                    qua.align()
                    qua.play('saturation'*qua.amp(drive_amp_scale), 'qubit')
                    qua.wait(readoutwait, 'resonator')
                    qua.measure('readout'*qua.amp(read_amp_scale), 'resonator', None,
                                qua.dual_demod.full(
                                    'cos', 'out1', 'sin', 'out2', I),
                                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                    qua.save(I, I_st)
                    qua.save(Q, Q_st)
                qua.save(n, n_st)

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(len(freqs)).average().save('I')
                Q_st.buffer(len(freqs)).average().save('Q')

        self.qmprog = prog
        return prog

    def _initialize_liveplot(self, ax):
        freqs = self.params['qubitIFs']
        line, = ax.plot(freqs/1e6, np.full(len(freqs), np.nan))
        ax.set_title("qubit spectroscopy analysis")
        ax.set_xlabel(f"drive IF [MHz] + {self.config['qubitLO']/1e9:f}GHz")
        ax.set_ylabel("|S|  (linear)")
        self.line = line

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.abs(res['Z']) /
                            self.config['readout_len'] * 2**12)
        ax.relim(), ax.autoscale(), ax.autoscale_view()


class QMQubitSpec_P2 (QMProgram):
    def __init__(self, qmm, config, Navg, qubitIFs, drive_amps):
        """
        qubitIFs in Hz
        driveamps in Volts (saturation_amp is ignored)
        """
        super().__init__(qmm, config)
        self.params = {
            'Navg': Navg,
            'qubitIFs': qubitIFs,
            'drive_amps': drive_amps}

    def _make_program(self):
        read_amp_scale = self.config['readout_amp'] / \
            self.config['qmconfig']['waveforms']['readout_wf']['sample']
        assert self.config['saturation_len'] == self.config['qmconfig']['pulses']['saturation_pulse']['length']
        assert self.config['readout_len'] == self.config['qmconfig']['pulses']['readout_pulse']['length']
        freqs = self.params['qubitIFs']

        self.params['drive_power'] = opx_amp2pow(self.params['drive_amps'], self.config['qubit_output_gain'])
        amps = self.params['drive_amps'] / \
            self.config['qmconfig']['waveforms']['saturation_wf']['sample']
        assert np.all(np.abs(self.params['drive_amps']) <=
                      0.5), "Output amplitudes need to be in range -0.5 to 0.5V."
        if np.any(amps < -8) or np.any(amps >= 8):
            raise ValueError(
                "Drive amplitudes cannot be scaled to target voltage because ratio is out of fixed value range [-8 to 8).")

        assert self.config['saturation_len'] >= self.config['readout_len']
        readoutwait = int(
            (self.config['saturation_len'] - self.config['readout_len']) / 2 / 4)  # cycles

        with qua.program() as prog:
            n = qua.declare(int)
            f = qua.declare(int)
            a = qua.declare(qua.fixed)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            n_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                with qua.for_(*from_array(f, freqs)):
                    qua.update_frequency('qubit', f)
                    with qua.for_(*from_array(a, amps)):
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        # randomize demod error
                        qua.wait(rand.rand_int(50)+4, 'resonator')
                        qua.align()
                        qua.play('saturation'*qua.amp(a), 'qubit')
                        qua.wait(readoutwait, 'resonator')
                        qua.measure('readout'*qua.amp(read_amp_scale), 'resonator', None,
                                    qua.dual_demod.full(
                                        'cos', 'out1', 'sin', 'out2', I),
                                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                qua.save(n, n_st)

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(len(freqs), len(amps)).average().save('I')
                Q_st.buffer(len(freqs), len(amps)).average().save('Q')

        self.qmprog = prog
        return prog

    def _initialize_liveplot(self, ax):
        freqs = self.params['qubitIFs']  # Hz
        amps = self.params['drive_amps']  # V
        # readoutpower = 10*np.log10(self.config['readout_amp']**2 * 10) # V to dBm
        drivepower = opx_amp2pow(amps, self.config['qubit_output_gain'])

        xx, yy = np.meshgrid(freqs/1e6, drivepower, indexing='ij')
        self.img = ax.pcolormesh(xx, yy, np.full(
            (len(freqs), len(amps)), np.nan), shading='nearest')
        self.colorbar = ax.get_figure().colorbar(self.img, ax=ax, orientation='horizontal')
        self.colorbar.set_label('arg S')

        ax.set_xlabel(
            f"drive IF / MHz + LO {self.config['qubitLO']/1e9:f} / GHz")
        ax.set_ylabel("drive power / dBm")
        axright = ax.secondary_yaxis(
            'right', functions=(
                lambda p: opx_pow2amp(p, self.config['qubit_output_gain']),
                lambda a: opx_amp2pow(a, self.config['qubit_output_gain'])))
        axright.set_ylabel('drive amplitude / V')

        readoutpower = opx_amp2pow(self.config['readout_amp'])
        ax.set_title(
            "qubit vs f2, P2\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit LO {self.config['qubitLO']/1e9:.3f}GHz"
            f"   Navg {self.params['Navg']}\n"
            f"{self.config['readout_len']/1e3:.0f}us readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.config['saturation_len']/1e3:.0f}us drive,  {self.config['qubit_output_gain']:+.1f}dB output gain",
            fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        # Note: setting an empty Normalize resets the colorscale
        Z = np.unwrap(np.unwrap(np.angle(res['Z']), axis=0))
        self.img.set(array=Z)
        self.img.autoscale()
        # ax.relim(), ax.autoscale(), ax.autoscale_view()
        
    def find_dip(self, good_power_estimate_dBm, window_length=5,
                 apply_to_config=None, ax=None, printinfo=True):
        """Tries to estimate qubit frequency by finding the minimum in the arg(Z) response.

        Raises PipelineException if no reliable dip is found.

        Sets qubit IF in config supplied to 'apply_to_config'.
        Returns dip IF frequency.
        """
        qubitIFs = self.params['qubitIFs']
        power = self.params['drive_power']
        res = self._retrieve_results()
        argZ = np.angle(res['Z'])

        if good_power_estimate_dBm < min(power) or good_power_estimate_dBm > max(power):
            warnings.warn(f"Good power estimate {good_power_estimate_dBm:+.1f}dBm is outside range of measured powers. Clipping to range.")
        ampi = np.argmin(np.abs(power - good_power_estimate_dBm))
        if window_length < 3:
            warnings.warn(f"Smoothing window {window_length} smaller 3. Skip smoothing.")
            filt = argZ[:,ampi]
        else:
            filt = savgol_filter(argZ[:,ampi], window_length, polyorder=2)
        qi = np.argmin(filt)
        signal = np.abs(filt[qi] - np.mean(filt))
        noise = np.std(np.diff(argZ[:,ampi]))/2**0.5
        if printinfo:
            print("Fine tune qubit IF")
            print(f"    {signal} signal vs {noise} noise level")
            print(f"    best IF: {qubitIFs[qi]/1e6}MHz")
        if signal < 3*noise:
            raise PipelineException(f"Signal {signal} to noise {noise} not larger 3.")
        if qi <= 1 or qi >= len(qubitIFs)-2:
            raise PipelineException(f"Minimum on the boundary of IF range {qi} in [0,{len(qubitIFs)-1}].")

        fq = qubitIFs[qi]
        if ax is not None:
            ax.plot([fq/1e6], [power[ampi]], '.', color='r')
        if apply_to_config is not None:
            apply_to_config['qubitIF'] = fq
        return fq


class QMReadoutSNR (QMProgram):
    """Uses short readout pulse and saturation pulse."""

    def __init__(self, qmm, config, Navg, resonatorIFs, readout_amps, drive_len):
        pass
        super().__init__(qmm, config)
        self.params = {
            'Navg': Navg,
            'resonatorIFs': resonatorIFs,
            'readout_amps': readout_amps,
            'drive_len': drive_len}

    def _make_program(self):
        drive_amp_scale = self.config['saturation_amp'] / \
            self.config['qmconfig']['waveforms']['saturation_wf']['sample']
        freqs = self.params['resonatorIFs']

        assert np.all(np.abs(self.params['readout_amps']) <= 0.5), "Output amplitudes need to be in range -0.5 to 0.5V."
        amps = self.params['readout_amps'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']
        if np.any(amps < -2) or np.any(amps >= 2):
            raise ValueError(
                "Readout amplitudes cannot be scaled to target voltage because ratio is out of scaling range [-2 to 2].")

        assert self.params['drive_len'] % 4 == 0
        drive_len_cycles = self.params['drive_len'] // 4
        readoutwait_cycles = drive_len_cycles - self.config['short_readout_len']//4
        assert readoutwait_cycles >= 4

        with qua.program() as prog:
            n = qua.declare(int)
            f = qua.declare(int)
            a = qua.declare(qua.fixed)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            n_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('qubit', self.config['qubitIF'])
            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                with qua.for_(*from_array(f, freqs)):
                    qua.update_frequency('resonator', f)
                    with qua.for_(*from_array(a, amps)):
                        # drive OFF
                        qua.measure(
                            'short_readout'*qua.amp(a), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator') # randomize demod error

                        # drive ON
                        qua.align()
                        qua.play('saturation'*qua.amp(drive_amp_scale), 'qubit')
                        qua.wait(readoutwait_cycles, 'resonator')
                        # qua.align()
                        qua.measure(
                            'short_readout'*qua.amp(a), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator') # randomize demod error
                qua.save(n, n_st)

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(len(freqs), len(amps), 2).average().save('I')
                Q_st.buffer(len(freqs), len(amps), 2).average().save('Q')

        self.qmprog = prog
        return prog

    def _initialize_liveplot(self, ax):
        freqs = self.params['resonatorIFs']  # Hz
        amps = self.params['readout_amps']  # V
        power = opx_amp2pow(amps, self.config['resonator_output_gain'])

        xx, yy = np.meshgrid(freqs/1e6, power, indexing='ij')
        self.img = ax.pcolormesh(xx, yy, np.full(
            (len(freqs), len(amps)), np.nan), shading='nearest')
        ax.set_xlabel("resonator IF / MHz")
        ax.set_ylabel("readout power / dBm")
        axright = ax.secondary_yaxis(
            'right', functions=(
                lambda p: opx_pow2amp(p, self.config['resonator_output_gain']),
                lambda a: opx_amp2pow(a, self.config['resonator_output_gain'])))
        axright.set_ylabel('readout amplitude / V')

        drivepower = opx_amp2pow(self.config['saturation_amp'])
        ax.set_title(
            f"readout SNR,  Navg {self.params['Navg']}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.3f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout,  {self.config['resonator_output_gain']:+.1f}dB output gain\n"
            f"{self.params['drive_len']:.0f}ns drive,  {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB",
            fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        dist = np.abs(res['Z'][...,1] - res['Z'][...,0])# / self.params['readout_amps'][None,:]
        self.img.set_array(dist)
        self.img.autoscale()


class QMTimeRabi (QMProgram):
    """Uses short readout pulse and saturation pulse."""

    def __init__(self, qmm, config, Navg, max_duration_ns, drive_read_overlap_cycles=0):
        super().__init__(qmm, config)
        self.params = {
            'Navg': Navg,
            'max_duration_ns': max_duration_ns,
            'drive_read_overlap_cycles': drive_read_overlap_cycles}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']
        drive_amp_scale = self.config['saturation_amp'] / \
            self.config['qmconfig']['waveforms']['saturation_wf']['sample']

        overlap_clk = self.params['drive_read_overlap_cycles']
        assert type(overlap_clk) is int and overlap_clk >= 0

        # minimum duration: 4cycles = 16ns
        maxduration = self.params['max_duration_ns']
        if maxduration < 16 or maxduration % 4 != 0:
            raise ValueError("Max duration of rabi pulse needs to be at least 16ns and a multiple of 4.")
        maxduration_clk = maxduration // 4
        duration_clk = np.arange(4, maxduration_clk, 1)
        self.params['duration_ns'] = np.arange(0, maxduration_clk*4, 1)

        # Waveforms are padded to a multiple of 4 samples and a minimum length of 16 samples
        # (with padding added as zeros at the beginning).
        # Remember: baking modifies the qmconfig but this class instance uses its own deep-copy.
        with baking(self.config['qmconfig'], padding_method='left') as bake0:
            bake0.add_op('drive_0', 'qubit', [[0], [0]])
            bake0.play('drive_0', 'qubit')
        baked_saturation = [bake0]
        for l in range(1, 16):
            with baking(self.config['qmconfig'], padding_method='left') as bake:
                bake.add_op('drive_%d'%l, 'qubit', [[self.config['saturation_amp']]*l, [0]*l])
                bake.play('drive_%d'%l, 'qubit')
            baked_saturation.append(bake)

        with qua.program() as prog:
            n = qua.declare(int)
            t = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            n_st = qua.declare_stream()
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            qua.update_frequency('qubit', self.config['qubitIF'])
            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                qua.save(n, n_st)
                for l in range(16):
                    qua.align()
                    qua.wait(12+overlap_clk, 'qubit')
                    baked_saturation[l].run()
                    qua.wait(16, 'resonator')
                    qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                    qua.save(I, I_st)
                    qua.save(Q, Q_st)
                    qua.wait(self.config['cooldown_clk'], 'resonator')
                    qua.wait(rand.rand_int(50)+4, 'resonator') # randomize demod error

                with qua.for_(*from_array(t, duration_clk)):
                    for i in range(4):
                        qua.align()
                        qua.wait(12+overlap_clk, 'qubit')
                        baked_saturation[i].run()
                        qua.play('saturation'*qua.amp(drive_amp_scale), 'qubit', duration=t)
                        qua.wait(16, 'resonator')
                        qua.wait(t, 'resonator')
                        qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator')

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(len(duration_clk)*4+16).average().save('I')
                Q_st.buffer(len(duration_clk)*4+16).average().save('Q')

        self.qmprog = prog
        return prog

    def check_timing(self, duration_cycles=25000, plot=True):
        """Simulate program and check alignment of drive and readout alignment.
        Raises PipelineException if the alignment is not constant for all pulses.
        Returns the alignment timing in ns.
        """
        if not hasattr(self, 'qmprog'):
            self._make_program()
        simulate_config = SimulationConfig(duration=duration_cycles)
        job = self.qmm.simulate(self.config['qmconfig'], self.qmprog, simulate_config)

        if plot:
            plt.figure()
            job.get_simulated_samples().con1.plot()

        analog = job.get_simulated_samples().con1.analog
        drive = (analog['3'] - analog['3'][0]) + 1j * (analog['4'] - analog['4'][0])
        read = (analog['7'] - analog['7'][0]) + 1j * (analog['8'] - analog['8'][0])
        # end of drive pulses
        drivestop = np.nonzero(drive)[0][:-1][np.diff(np.nonzero(drive)[0]) > 1]
        # start of readout pulses
        readstart = np.nonzero(read)[0][1:][np.diff(np.nonzero(read)[0]) > 1]
        if plot:
            plt.scatter(drivestop, [0]*drivestop.size)
            plt.scatter(readstart, [0]*readstart.size, color='C2')

        l = min(readstart.size, drivestop.size)
        overlap = drivestop[:l] - readstart[:l]
        print("Waveform overlap:", overlap, "ns")
        if not np.all(overlap == overlap[0]):
            raise PipelineException("Drive/readout waveform alignment not constant.")
        return overlap

    def _initialize_liveplot(self, ax):
        durations = self.params['duration_ns']
        self.line, = ax.plot(durations, np.full(len(durations), np.nan))
        ax.set_xlabel("drive duration / ns")
        ax.set_ylabel("arg S")
        readoutpower = opx_amp2pow(self.config['readout_amp'])
        drivepower = opx_amp2pow(self.config['saturation_amp'])
        ax.set_title(
            f"Time Rabi,   Navg {self.params['Navg']:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.0f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB,"
            f"  {self.params['drive_read_overlap_cycles']//4}ns overlap",
            fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.unwrap(np.angle(res['Z'])))
        ax.relim(), ax.autoscale(), ax.autoscale_view()


class QMTimeRabiChevrons (QMTimeRabi):
    """Uses short readout pulse and saturation pulse.
    Inherits from QMTimeRabi for check_timing()
    """

    def __init__(self, qmm, config, Navg, qubitIFs, max_duration_ns, drive_read_overlap_cycles=0):
        super().__init__(qmm, config, None, None)
        self.params = {
            'Navg': Navg,
            'qubitIFs': qubitIFs,
            'max_duration_ns': max_duration_ns,
            'drive_read_overlap_cycles': drive_read_overlap_cycles}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']
        drive_amp_scale = self.config['saturation_amp'] / \
            self.config['qmconfig']['waveforms']['saturation_wf']['sample']

        qubitIFs = self.params['qubitIFs']
        overlap_clk = self.params['drive_read_overlap_cycles']
        assert type(overlap_clk) is int and overlap_clk >= 0

        # minimum duration: 4cycles = 16ns
        maxduration = self.params['max_duration_ns']
        if maxduration < 16 or maxduration % 4 != 0:
            raise ValueError("Max duration of rabi pulse needs to be at least 16ns and a multiple of 4.")
        maxduration_clk = maxduration // 4
        duration_clk = np.arange(4, maxduration_clk, 1)
        self.params['duration_ns'] = np.arange(0, maxduration_clk*4, 1)

        # Waveforms are padded to a multiple of 4 samples and a minimum length of 16 samples
        # (with padding added as zeros at the beginning).
        # Remember: baking modifies the qmconfig but this class instance uses its own deep-copy.
        with baking(self.config['qmconfig'], padding_method='left') as bake0:
            bake0.add_op('drive_0', 'qubit', [[0], [0]])
            bake0.play('drive_0', 'qubit')
        baked_saturation = [bake0]
        for l in range(1, 16):
            with baking(self.config['qmconfig'], padding_method='left') as bake:
                bake.add_op('drive_%d'%l, 'qubit', [[self.config['saturation_amp']]*l, [0]*l])
                bake.play('drive_%d'%l, 'qubit')
            baked_saturation.append(bake)

        with qua.program() as prog:
            n = qua.declare(int)
            f = qua.declare(int)
            t = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            n_st = qua.declare_stream()
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                qua.save(n, n_st)
                with qua.for_(*from_array(f, qubitIFs)):
                    qua.update_frequency('qubit', f)
                    for l in range(16):
                        qua.align()
                        qua.wait(12+overlap_clk, 'qubit')
                        baked_saturation[l].run()
                        qua.wait(16, 'resonator')
                        qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator') # randomize demod error

                    with qua.for_(*from_array(t, duration_clk)):
                        for i in range(4):
                            qua.align()
                            qua.wait(12+overlap_clk, 'qubit')
                            baked_saturation[i].run()
                            qua.play('saturation'*qua.amp(drive_amp_scale), 'qubit', duration=t)
                            qua.wait(16, 'resonator')
                            qua.wait(t, 'resonator')
                            qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                            qua.save(I, I_st)
                            qua.save(Q, Q_st)
                            qua.wait(self.config['cooldown_clk'], 'resonator')
                            qua.wait(rand.rand_int(50)+4, 'resonator')

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(len(qubitIFs), len(duration_clk)*4+16).average().save('I')
                Q_st.buffer(len(qubitIFs), len(duration_clk)*4+16).average().save('Q')

        self.qmprog = prog
        return prog

    def _figtitle(self, Navg):
        readoutpower = opx_amp2pow(self.config['readout_amp'])
        drivepower = opx_amp2pow(self.config['saturation_amp'])
        return (
            f"Rabi Chevrons,   Navg {Navg:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit LO {self.config['qubitLO']/1e9:.3f}GHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB,"
            f"  {self.params['drive_read_overlap_cycles']//4}ns overlap")

    def _initialize_liveplot(self, ax):
        durations = self.params['duration_ns']
        freqs = self.params['qubitIFs']

        xx, yy = np.meshgrid(durations, freqs/1e6, indexing='ij')
        self.img = ax.pcolormesh(xx, yy, np.full(
            (len(durations), len(freqs)), np.nan), shading='nearest')
        self.colorbar = ax.get_figure().colorbar(self.img, ax=ax, orientation='horizontal')
        self.colorbar.set_label("arg S")
        ax.set_ylabel("drive IF / MHz")
        ax.set_xlabel("drive duration / ns")
        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.img.set_array(np.unwrap(np.unwrap(np.angle(res['Z'])), axis=0).T)
        self.img.autoscale()
        self.ax.set_title(self._figtitle(res['iteration']+1), fontsize=8)


class QMPowerRabi (QMProgram):
    """Uses short readout pulse and saturation pulse."""

    def __init__(self, qmm, config, Navg, duration_ns, drive_amps, drive_read_overlap_cycles=0):
        super().__init__(qmm, config)
        self.params = {
            'Navg': Navg,
            'duration_ns': duration_ns,
            'drive_amps': drive_amps,
            'drive_read_overlap_cycles': drive_read_overlap_cycles}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']
        drive_amp_scale = self.params['drive_amps'] / 0.316
        self.params['drive_power'] = opx_amp2pow(self.params['drive_amps'], self.config['qubit_output_gain'])

        overlap_clk = self.params['drive_read_overlap_cycles']
        assert type(overlap_clk) is int and overlap_clk >= 0

        duration = self.params['duration_ns']

        # Waveforms are padded to a multiple of 4 samples and a minimum length of 16 samples
        # (with padding added as zeros at the beginning).
        # Remember: baking modifies the qmconfig but this class instance uses its own deep-copy.
        with baking(self.config['qmconfig'], padding_method='left') as bakedrive:
            bakedrive.add_op('drive_0', 'qubit', [[0.316]*duration, [0]*duration])
            bakedrive.play('drive_0', 'qubit')
        drivepulselen_cycles = int(np.ceil(duration/4))

        with qua.program() as prog:
            n = qua.declare(int)
            a = qua.declare(qua.fixed)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            n_st = qua.declare_stream()
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            qua.update_frequency('qubit', self.config['qubitIF'])
            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                qua.save(n, n_st)
                with qua.for_(*from_array(a, drive_amp_scale)):
                    qua.wait(12+overlap_clk, 'qubit')
                    bakedrive.run(amp_array=[('qubit', a)])
                    qua.wait(12+drivepulselen_cycles, 'resonator')
                    qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                    qua.save(I, I_st)
                    qua.save(Q, Q_st)
                    qua.wait(self.config['cooldown_clk'], 'resonator')
                    qua.wait(rand.rand_int(50)+4, 'resonator')

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(len(drive_amp_scale)).average().save('I')
                Q_st.buffer(len(drive_amp_scale)).average().save('Q')

        self.qmprog = prog
        return prog

    def _initialize_liveplot(self, ax):
        amps = self.params['drive_amps']
        self.line, = ax.plot(amps, np.full(len(amps), np.nan))
        ax.set_xlabel("drive amplitude / V")
        ax.set_ylabel("arg S")
        readoutpower = opx_amp2pow(self.config['readout_amp'])
        ax.set_title(
            f"Power Rabi,   Navg {self.params['Navg']:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.0f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.params['duration_ns']:.0f}ns drive,  {self.config['qubit_output_gain']:+.1f}dB output gain\n"
            f"{self.params['drive_read_overlap_cycles']//4}ns overlap",
            fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.unwrap(np.angle(res['Z'])))
        ax.relim(), ax.autoscale(), ax.autoscale_view()

# %%

if __name__ == '__main__':
    import importlib
    import configuration as config
    importlib.reload(config)

    qmm = qminit.connect()
    # QMMixerCalibration(qmm, config).run()
    # p = QMTimeOfFlight(qmm, config, Navg=100)
    # results = p.run()
    # print(results.keys())

    config.readout_amp = 0.0316
    p = QMResonatorSpec(qmm, config, Navg=500,
                        resonatorIFs=np.arange(202e6, 212e6, 0.05e6))
    results = p.run(plot=True)
    
    config.resonator_output_gain = 10
    p = QMResonatorSpec_P2(
        qmm, config, Navg=100,
        resonatorIFs=np.arange(203e6, 209e6, 0.1e6),
        readout_amps=np.logspace(np.log10(0.000316), np.log10(0.0316), 21))
    results = p.run(plot=True)
    config.resonator_output_gain = -20

    config.saturation_amp = 0.1
    p = QMQubitSpec(qmm, config, Navg=1000,
                    qubitIFs=np.arange(50e6, 450e6, 5e6))
    results = p.run(plot=True)

    p = QMQubitSpec_P2(
        qmm, config, Navg=500, qubitIFs=np.arange(50e6, 450e6, 5e6),
        drive_amps=np.logspace(np.log10(0.01), np.log10(0.316), 11))
    results = p.run(plot=True)
