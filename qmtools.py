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

Use a dict ('config') with settings, including octave settings.
It contains config['qmconfig'] which is the quantum machine config.
Settings like qubit IF might differ between config['qubitIF'] and config['qmconfig'][...],
in which case it will be updated during run time to match qubit['qubitIF'].
This allows reusing the same quantum machine (config['qmconfig'] not changed) (Not implemented yet).

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

TODO
====

- Result dict fields may be None (QM ResultStreams with no data) or non-existent (not calculated because no data)
  This should be unified: Only save fields where ResultStream is not None.
  Need to fix this in all occurrences: res[key] is None -> key in res
- How to treat correctly results from secondary functions working on results
  (that fit data / find optimal parameters).
- How to make secondary functions plot nicely in same axes
"""

# TODO rename "saturation" pulse to "drive" because we use it for all drive pulses, including coherent ones

import time
import inspect
import warnings
from copy import deepcopy
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from datetime import datetime

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

    def simulate(self, duration_cycles, plot=True):
        if not hasattr(self, 'qmprog'):
            self._make_program()
        simulate_config = SimulationConfig(duration=duration_cycles)
        job = self.qmm.simulate(self.config['qmconfig'], self.qmprog, simulate_config)

        if plot:
            plt.figure()
            job.get_simulated_samples().con1.plot()

        return job

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

        # If LO is different for two elements on the same mixer, the wrong
        # calibration is loaded.
        assert (self.config['qmconfig']['elements']['qubit']['mixInputs']['lo_frequency']
                == self.config['qmconfig']['elements']['qubit2']['mixInputs']['lo_frequency']), \
            "Elements qubit and qubit2 do not have same LO freq"

        # qmm.open_qm modifies the mixer section of the configuration dict,
        # replacing it with values from the calibration db
        qm = self.qm = self.qmm.open_qm(self.config['qmconfig'])
        self._init_octave(qm)
        # TODO: precompile, but then needs to run on same qm
        # progid = self.compile()
        # pendingjob = qm.queue.add_compiled(progid)
        # job = pendingjob.wait_for_execution()

        job = self.last_job = qm.execute(self.qmprog)
        resulthandles = job.result_handles
        tstart = self.last_tstart = time.time()

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
        Niter = self.params.get('Niter', self.params.get('Navg', np.nan))
        try:
            while resulthandles.is_processing():
                if hasiter and printn:
                    iteration = resulthandles.iteration.fetch_all() or 1
                    print(
                        f"iteration={iteration}, remaining: {(Niter-iteration) * (time.time()-tstart)/iteration:.0f}s")
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

        trun = self.last_trun = time.time() - tstart
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
        Reworks named fields of streams so that res['I'] are the values and
        res['I_timestamps'] are the associated timestamps if available.

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
        fetched = {k: (f.fetch_all() if len(f) else None) for k, f in resulthandles._all_results.items()}
        # Rework named fields
        res = {k: v for k, v in fetched.items() if v is None or v.dtype.names is None}
        res |= {k: v['value'] for k, v in fetched.items() if v is not None and v.dtype.names is not None and 'value' in v.dtype.names}
        res |= {k+'_timestamps': v['timestamp'] for k, v in fetched.items() if v is not None and v.dtype.names is not None and 'timestamp' in v.dtype.names}
        # Copy results because they are mutable and we don't want to contaminate
        # the original results in job.result_handles
        res = {k: (np.copy(a) if isinstance(a, np.ndarray) else a)
               for k, a in res.items()}
        # Magically prepare Z
        if 'I' in res and 'Q' in res and 'Z' not in res:
            if res['I'] is not None and res['Q'] is not None:
                try:
                    res['Z'] = (res['I'] + 1j * res['Q'])
                except ValueError:
                    # Probably the shapes didnt match yet
                    res['Z'] = None
            else:
                res['Z'] = None
        # Also save the time it took to run and start time (UTC timestamps)
        if hasattr(self, 'last_tstart'):
            res['job_starttime'] = self.last_tstart
        if hasattr(self, 'last_trun'):
            res['job_runtime'] = self.last_trun
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
        self.last_run = None

    def _make_program(self):
        pass

    def compile(self):
        raise NotImplementedError

    def run(self):
        self.last_run = time.time()

        qm = self.qmm.open_qm(self.config['qmconfig'])
        self._init_octave(qm)
        print("Running calibration on resonator channel...")
        rcal = qm.octave.calibrate_element(
            'resonator', [(self.config['resonatorLO'], self.config['resonatorIF'])])
        rcal = list(rcal.values())[0].__dict__
        # delete to remove MonitorData object from qua
        # Otherwise qua.octave is required to be installed to unpickle this
        del rcal['temperature']
        if rcal['correction'] == [1, 0, 0, 1]:
            warnings.warn(f"Resonator calibration LO {self.config['resonatorLO']/1e9:f}GHz IF{self.config['resonatorIF']/1e6:f}MHz failed (result is identity matrix).")

        qubitLOs = list(self.params['qubitLOs'])
        if self.config['qubitLO'] not in qubitLOs:
            qubitLOs.append(self.config['qubitLO'])
        print(f"Running calibration on qubit channel for {len(self.params['qubitLOs'])} LO frequencies...")
        qcal = []
        for lof in qubitLOs:
            qm.octave.set_lo_frequency('qubit', lof)
            cal = qm.octave.calibrate_element(
                'qubit', [(lof, self.config['qubitIF'])])
            cal = list(cal.values())[0].__dict__
            del cal['temperature']
            if cal['correction'] == [1, 0, 0, 1]:
                warnings.warn(f"Qubit calibration LO {lof/1e9:f}GHz IF{self.config['qubitIF']/1e6:f}MHz failed (result is identity matrix).")
            qcal.append(cal)
        return {'resonator': rcal, 'qubit': qcal} | self.params | {'config': self.config}

    def run_after_interval(self, interval):
        """Rerun calibration if more than 'interval' seconds have passed since
        the last call to this calibration instance."""
        if self.last_run is None or time.time() - self.last_run > interval:
            return self.run()
        else:
            print(f"Last calibration started at {datetime.fromtimestamp(self.last_run).isoformat()}. No recalibration.")
            return None


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
            print('    Single run noise:', np.std(tof['adc1_single_run']), np.std(tof['adc2_single_run']), 'ADC units')
            print('    Exp. avg. noise:', np.std(tof['adc1_single_run'])/np.sqrt(tof['Navg']), np.std(tof['adc2_single_run'])/np.sqrt(tof['Navg']), 'ADC units')
            #print('Averaged noise:', np.std(np.diff(tof['adc1']))/np.sqrt(2), np.std(np.diff(tof['adc2']))/np.sqrt(2))
            print('    Averaged noise:', np.std(tof['adc1']), np.std(tof['adc2']), 'ADC units')
            print('    Offset error:', offserr, 'ADC')
            print('    Offset error uncertainty:', np.std(tof['adc1'])/np.sqrt(nsamples), np.std(tof['adc2'])/np.sqrt(nsamples), 'ADC samples')
            print('    Offset correct to:', newoffs1, newoffs2, 'V')
        config['qmconfig']['controllers']['con1']['analog_inputs'][1]['offset'] = newoffs1
        config['qmconfig']['controllers']['con1']['analog_inputs'][2]['offset'] = newoffs2


class QMResonatorSpec (QMProgram):
    """Continuous-wave one-tone spectroscopy of readout resonator.

    Scales IQ to volts and corrects electrict delay.

    Has utility function to fit Lorentzian to extract resonance frequency.
    """
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
            freqs[np.argmax(np.abs(Z))], #(np.mean(freqs[:5]) + np.mean(freqs[-5:])) / 2,  # f0
            (np.max(freqs[-1])-np.min(freqs[0])) / 3,  # width
            np.max(np.abs(Z)),  # amplitude
            np.angle(Z[np.argmax(np.abs(Z))])  # angle
        ]
        print(p0)
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


class QMNoiseSpectrum (QMProgram):
    """Readout many times to later do FFT of it.
    Does not use the cooldown time.

    Will record Nsamples at the resonator frequency. The spacing between samples
    is the intrinsic program time plus wait_ns (minimum 16ns).

    The sample spacing determines the maximum frequency (which usually is way
    higher than of interest). The total length of the measurement T, determines
    the frequency resolution 1/T. The number of samples reduces the noise, since
    the FFT is basically averaging with a rotation.

    Usually need large number of samples to have a low noise floor, but this
    unnecessarily increases the saved output. Thus fcut_Hz can be used to save
    only a subset of the FFT in -fcut_Hz to fcut_Hz.

    If fcut_Hz is None the full raw data and FFT is saved, otherwise only the
    FFT inside the cutoff.
    """

    def __init__(self, qmm, config, Nsamples, wait_ns=16, fcut_Hz=None):
        super().__init__(qmm, config)
        self.params = {'Nsamples': Nsamples, 'wait_ns': wait_ns, 'fcut_Hz': fcut_Hz}

    def _make_program(self):
        amp_scale = self.config['readout_amp'] / \
            self.config['qmconfig']['waveforms']['readout_wf']['sample']
        assert self.config['readout_len'] == self.config['qmconfig']['pulses']['readout_pulse']['length']

        with qua.program() as prog:
            n = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            n_st = qua.declare_stream()

            qua.update_frequency('resonator', self.config['resonatorIF'])

            with qua.for_(n, 0, n < self.params['Nsamples'], n + 1):
                qua.measure('readout'*qua.amp(amp_scale), 'resonator', None,
                            qua.dual_demod.full(
                                'cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                qua.save(I, I_st)
                qua.save(Q, Q_st)
                qua.save(n, n_st)
                qua.wait(self.params['wait_ns']//4)

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.with_timestamps().save_all('I')
                Q_st.save_all('Q')

        self.qmprog = prog
        return prog

    def _retrieve_results(self, resulthandles=None):
        """Normalizes readout and calculates FFT with correct freq axis"""
        res = super()._retrieve_results(resulthandles)
        # Normalize
        nsamples = self.config['qmconfig']['pulses']['readout_pulse']['length']
        # Note: don't use (a *= 3) because it modifies the result in the iterator.
        if res['I'] is not None:
            res['I'] = res['I'] * (2**12 / nsamples)
        if res['Q'] is not None:
            res['Q'] = res['Q'] * (2**12 / nsamples)
        if res['Z'] is not None:
            res['Z'] = res['Z'] * (2**12 / nsamples)
            # Calculated FFT
            res['fft'] = np.fft.fftshift(np.fft.fft(res['Z'] - np.mean(res['Z']))) / res['Z'].size
            # Note: I always has same shape as I_timestamps
            # So Z is guaranteed to have same shape as t
            t = res['I_timestamps'] * 1e-9
            dt = (np.max(t) - np.min(t)) / (t.size-1)
            res['fftfreq'] = np.fft.fftshift(np.fft.fftfreq(t.size, dt))

            if self.params['fcut_Hz'] is not None:
                fcut = self.params['fcut_Hz']
                # cut FFT
                mask = (res['fftfreq'] >= -fcut) & (res['fftfreq'] <= fcut)
                res['fft'] = res['fft'][mask]
                res['fftfreq'] = res['fftfreq'][mask]
                # clean large data
                del res['I'], res['Q'], res['Z'], res['I_timestamps']

        return res

    def _initialize_liveplot(self, ax):
        readoutpower = opx_amp2pow(self.config['readout_amp'])
        line, = ax.plot([np.nan], [np.nan])
        ax.set_xlabel("Frequency / Hz")
        ax.set_ylabel("|FFT S|")
        ax.set_title(
            f"resonator noise spectrum   {self.params['Nsamples']} samples\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"{self.config['readout_len']/1e3:.0f}us readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB",
            fontsize=8)
        self.line = line
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if 'fft' not in res:
            return
        self.line.set_data(res['fftfreq'], np.abs(res['fft']))
        ax.relim(), ax.autoscale(), ax.autoscale_view()


class QMQubitSpec (QMProgram):
    """Continuous wave two-tone spectroscopy of qubit.

    Uses 'saturation' pulse (amplitude and qubitIF may be modified outside config.qmconfig),
    and 'readout' pulse (amplitude and resonatorIF may be modified outside config.qmconfig).

    Assumes saturation pulse longer than readout pulse.
    Readout is centered in saturation pulse time-wise.
    """
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
        freqs = self.params['qubitIFs'] + self.config['qubitLO']
        line, = ax.plot(freqs/1e9, np.full(len(freqs), np.nan))
        ax.set_title("qubit spectroscopy analysis")
        readoutpower = opx_amp2pow(self.config['readout_amp'])
        drivepower = opx_amp2pow(self.config['saturation_amp'])
        ax.set_title(
            "Qubit spectroscopy\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit LO {self.config['qubitLO']/1e9:.3f}GHz"
            f"   Navg {self.params['Navg']}\n"
            f"{self.config['readout_len']/1e3:.0f}us readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.config['saturation_len']/1e3:.0f}us drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB",
            fontsize=8)
        ax.set_xlabel(f"drive / GHz")
        ax.set_ylabel("arg S")
        self.line = line
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        # self.line.set_ydata(np.abs(res['Z']) /
        #                     self.config['readout_len'] * 2**12)
        self.line.set_ydata(np.unwrap(np.angle(res['Z'])))
        ax.relim(), ax.autoscale(), ax.autoscale_view()

    def find_dip(self, window_length=5, ax=None, printinfo=True):
        """Tries to estimate qubit frequency by finding the minimum in the arg(Z) response.

        Raises PipelineException if no reliable dip is found.

        Sets qubit IF in config supplied to 'apply_to_config'.
        Returns dip IF frequency.
        """
        qubitIFs = self.params['qubitIFs']
        res = self._retrieve_results()
        argZ = np.unwrap(np.angle(res['Z']))

        if window_length < 3:
            warnings.warn(f"Smoothing window {window_length} smaller 3. Skip smoothing.")
            filt = argZ
        else:
            filt = savgol_filter(argZ, window_length, polyorder=2)
        qi = np.argmin(filt)
        signal = np.abs(filt[qi] - np.median(filt))
        noise = np.std(np.diff(argZ))/2**0.5
        if printinfo:
            print("Fine tune qubit IF")
            print(f"    {signal/noise:.1e} SNR: {signal:.2e} signal vs {noise:.2e} noise level")
            print(f"    dip IF: {qubitIFs[qi]/1e6}MHz")
        if signal < 3*noise:
            raise PipelineException(f"Signal {signal} to noise {noise} not larger 3.")
        if qi <= 1 or qi >= len(qubitIFs)-2:
            raise PipelineException(f"Minimum on the boundary of IF range {qi} in [0,{len(qubitIFs)-1}].")

        fq = qubitIFs[qi]
        if ax is not None:
            ax.plot([(fq+self.config['qubitLO'])/1e9], [argZ[qi]], '.', color='r')
        return fq


class QMQubitSpecThreeTone (QMProgram):
    """Continuous wave drive constant at qubitIF then sweep another drive tone.

    Uses 'saturation' pulse (amplitude and qubitIF may be modified outside config.qmconfig),
    and 'readout' pulse (amplitude and resonatorIF may be modified outside config.qmconfig).
    """
    def __init__(self, qmm, config, Navg, third_amp, thirdIFs):
        super().__init__(qmm, config)
        self.params = {'thirdIFs': thirdIFs, 'third_amp': third_amp, 'Navg': Navg}

    def _make_program(self):
        read_amp_scale = self.config['readout_amp'] / \
            self.config['qmconfig']['waveforms']['readout_wf']['sample']
        drive_amp_scale = self.config['saturation_amp'] / \
            self.config['qmconfig']['waveforms']['saturation_wf']['sample']
        third_amp_scale = self.params['third_amp'] / \
            self.config['qmconfig']['waveforms']['saturation_wf']['sample']
        assert self.config['saturation_len'] == self.config['qmconfig']['pulses']['saturation_pulse']['length']
        assert self.config['readout_len'] == self.config['qmconfig']['pulses']['readout_pulse']['length']
        freqs = self.params['thirdIFs']

        # center readout in saturation pulse
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
            qua.update_frequency('qubit', self.config['qubitIF'])
            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                with qua.for_(*from_array(f, freqs)):
                    qua.update_frequency('qubit2', f)
                    qua.wait(self.config['cooldown_clk'], 'resonator')
                    # randomize demod error
                    qua.wait(rand.rand_int(50)+4, 'resonator')
                    qua.align()
                    qua.play('saturation'*qua.amp(drive_amp_scale), 'qubit')
                    qua.play('saturation'*qua.amp(third_amp_scale), 'qubit2')
                    qua.wait(readoutwait, 'resonator')
                    qua.measure('readout'*qua.amp(read_amp_scale), 'resonator', None,
                                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
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
        freqs = self.params['thirdIFs']
        self.line, = ax.plot(freqs/1e6, np.full(len(freqs), np.nan))

        drivepower = opx_amp2pow(self.config['saturation_amp'])
        thirdpower = opx_amp2pow(self.params['third_amp'])
        readoutpower = opx_amp2pow(self.config['readout_amp'])
        ax.set_title(
            f"Three-tone spectroscopy   Navg {self.params['Navg']}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.3f}MHz\n"
            f"{self.config['readout_len']/1e3:.0f}us readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.config['saturation_len']/1e3:.0f}us drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB\n"
            f"{self.config['saturation_len']/1e3:.0f}us third tone at {thirdpower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB",
            fontsize=8)
        ax.set_xlabel("third tone IF / MHz")
        ax.set_ylabel("arg S")
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.unwrap(np.angle(res['Z'])))
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
        self.colorbar = ax.get_figure().colorbar(self.img, ax=ax, orientation='horizontal', shrink=0.9)
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
    """Uses short readout pulse and saturation pulse.
    
    Need qubitIF to be tuned correctly to drive qubit into mixed state.

    Note: 4.28 fixed point numbers have a range of [-8 to 8) with a precision of 2^-28 = 3.7e-9.
    This means the typical demod result squared, 1e-5**2=1e-10 is smaller than the precision!
    Thus we cannot calculate the variance with a naive algorithm on the QM.

    For this reason the variance is computed from single shot results afterwards.
    """

    def __init__(self, qmm, config, Navg, resonatorIFs, readout_amps, drive_len, Nvar=1000):
        super().__init__(qmm, config)
        assert Nvar <= Navg, "Number of shots for variance needs to be equal or smaller than total samples used for average."
        self.params = {
            'Navg': Navg, 'Nvar': Nvar,
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
        #drive_len_cycles = self.params['drive_len'] // 4
        #readoutwait_cycles = drive_len_cycles - self.config['short_readout_len']//4
        #assert readoutwait_cycles >= 4

        with qua.program() as prog:
            n = qua.declare(int)
            f = qua.declare(int)
            a = qua.declare(qua.fixed)
            I, Q = qua.declare(qua.fixed), qua.declare(qua.fixed)
            I_st, Q_st = qua.declare_stream(), qua.declare_stream()
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
                        # qua.wait(readoutwait_cycles, 'resonator')
                        qua.align()
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
                I_st.buffer(self.params['Nvar'], len(freqs), len(amps), 2).save('I_single_shot')
                Q_st.buffer(self.params['Nvar'], len(freqs), len(amps), 2).save('Q_single_shot')

        self.qmprog = prog
        return prog

    def _figtitle(self, Navg):
        drivepower = opx_amp2pow(self.config['saturation_amp'])
        return (
            f"readout SNR,  Navg {Navg:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.3f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout,  {self.config['resonator_output_gain']:+.1f}dB output gain\n"
            f"{self.params['drive_len']:.0f}ns drive,  {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB"
        )

    def _retrieve_results(self, resulthandles=None):
        """Calculates signal variance and amplitude SNR.

        Returns dict with keys

        Z : complex array of shape (len(freqs), len(amps), 2)
            Signal mean, last axis is [drive OFF, drive ON].
        Zvar : complex array of shape (len(freqs), len(amps), 2)
            Signal variance.
            Only present if program ran long enough.
        SNR : real and positive array of shape (len(freqs), len(amps))
            Amplitude SNR where signal given by difference between ON/OFF
            measurement.
            Only present if program ran long enough.

        readout_power : array of shape (len(amps),)
            Readout amplitude converted to output power,
            including octave output gain.
        """
        res = super()._retrieve_results(resulthandles)
        if 'Z' in res:
            if 'I' in res: del res['I']
            if 'Q' in res: del res['Q']
        # Take variance of single shots and delete single shot data
        if res['I_single_shot'] is not None:
            res['Ivar'] = np.var(res['I_single_shot'], axis=0)
            del res['I_single_shot']
        if res['Q_single_shot'] is not None:
            res['Qvar'] = np.var(res['Q_single_shot'], axis=0)
            del res['Q_single_shot']
        # Calculate Zvar and SNR
        if 'Ivar' in res and 'Qvar' in res:
            res['Zvar'] = res['Ivar'] + 1j*res['Qvar']
            signal = np.abs(res['Z'][...,1] - res['Z'][...,0])
            noise = np.sqrt(np.mean(res['Ivar'], axis=-1) + np.mean(res['Qvar'], axis=-1))
            res['SNR'] = signal / noise
            del res['Ivar']
            del res['Qvar']
        res['readout_power'] = opx_amp2pow(self.params['readout_amps'], self.config['resonator_output_gain'])
        return res

    def _initialize_liveplot(self, ax):
        freqs = self.params['resonatorIFs']  # Hz
        amps = self.params['readout_amps']  # V
        power = opx_amp2pow(amps, self.config['resonator_output_gain'])

        xx, yy = np.meshgrid(freqs/1e6, power, indexing='ij')
        self.img = ax.pcolormesh(xx, yy, np.full(
            (len(freqs), len(amps)), np.nan), shading='nearest')
        self.colorbar = ax.get_figure().colorbar(self.img, ax=ax, orientation='horizontal', shrink=0.9)
        self.colorbar.set_label("|SON - SOFF|")
        ax.set_xlabel("resonator IF / MHz")
        ax.set_ylabel("readout power / dBm")
        axright = ax.secondary_yaxis(
            'right', functions=(
                lambda p: opx_pow2amp(p, self.config['resonator_output_gain']),
                lambda a: opx_amp2pow(a, self.config['resonator_output_gain'])))
        axright.set_ylabel('readout amplitude / V')

        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if 'SNR' in res:
            self.img.set_array(res['SNR'])
            self.colorbar.set_label('SNR')
        elif res['Z'] is not None:
            dist = np.abs(res['Z'][...,1] - res['Z'][...,0])
            self.img.set_array(dist)
        self.img.autoscale()
        ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)


class QMReadoutSNR_P1 (QMProgram):
    """Uses short readout pulse and saturation pulse.

    Need qubitIF to be tuned correctly to drive qubit into mixed state.
    """

    def __init__(self, qmm, config, Navg, readout_amps, drive_len, Nvar=1000):
        super().__init__(qmm, config)
        assert Nvar <= Navg
        self.params = {
            'Navg': Navg, 'Nvar': Nvar,
            'readout_amps': readout_amps,
            'drive_len': drive_len}

    def _make_program(self):
        drive_amp_scale = self.config['saturation_amp'] / \
            self.config['qmconfig']['waveforms']['saturation_wf']['sample']

        assert np.all(np.abs(self.params['readout_amps']) <= 0.5), "Output amplitudes need to be in range -0.5 to 0.5V."
        amps = self.params['readout_amps'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']
        if np.any(amps < -2) or np.any(amps >= 2):
            raise ValueError(
                "Readout amplitudes cannot be scaled to target voltage because ratio is out of scaling range [-2 to 2].")

        assert self.params['drive_len'] % 4 == 0
        #drive_len_cycles = self.params['drive_len'] // 4
        #readoutwait_cycles = drive_len_cycles - self.config['short_readout_len']//4
        #assert readoutwait_cycles >= 4

        with qua.program() as prog:
            n = qua.declare(int)
            a = qua.declare(qua.fixed)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            n_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('qubit', self.config['qubitIF'])
            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
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
                    # qua.wait(readoutwait_cycles, 'resonator')
                    qua.align()
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
                I_st.buffer(len(amps), 2).average().save('I')
                Q_st.buffer(len(amps), 2).average().save('Q')
                I_st.buffer(self.params['Nvar'], len(amps), 2).save('I_single_shot')
                Q_st.buffer(self.params['Nvar'], len(amps), 2).save('Q_single_shot')

        self.qmprog = prog
        return prog

    def _retrieve_results(self, resulthandles=None):
        """Calculates variance and SNR"""
        res = super()._retrieve_results(resulthandles)
        if 'Z' in res:
            if 'I' in res: del res['I']
            if 'Q' in res: del res['Q']
        # Take variance of single shots and delete single shot data
        if res['I_single_shot'] is not None:
            res['Ivar'] = np.var(res['I_single_shot'], axis=0)
            del res['I_single_shot']
        if res['Q_single_shot'] is not None:
            res['Qvar'] = np.var(res['Q_single_shot'], axis=0)
            del res['Q_single_shot']
        # Calculate Zvar and SNR
        if 'Ivar' in res and 'Qvar' in res:
            res['Zvar'] = res['Ivar'] + 1j*res['Qvar']
            signal = np.abs(res['Z'][...,1] - res['Z'][...,0])
            noise = np.sqrt(np.mean(res['Ivar'], axis=-1) + np.mean(res['Qvar'], axis=-1))
            res['SNR'] = signal / noise
            del res['Ivar']
            del res['Qvar']
        res['readout_power'] = opx_amp2pow(self.params['readout_amps'], self.config['resonator_output_gain'])
        return res

    def _figtitle(self, Navg):
        drivepower = opx_amp2pow(self.config['saturation_amp'])
        return (
            f"readout SNR,  Navg {Navg:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.3f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout,  {self.config['resonator_output_gain']:+.1f}dB output gain\n"
            f"{self.params['drive_len']:.0f}ns drive,  {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB"
        )

    def _initialize_liveplot(self, ax):
        amps = self.params['readout_amps']  # V
        power = opx_amp2pow(amps, self.config['resonator_output_gain'])
        self.line, = ax.plot(power, np.full(power.size, np.nan))
        ax.set_xlabel("readout power / dBm")
        ax.set_ylabel("|SON-SOFF|")
        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if 'SNR' in res:
            self.line.set_ydata(res['SNR'])
            self.ax.set_ylabel('SNR')
        elif res['Z'] is not None:
            dist = np.abs(res['Z'][:,1] - res['Z'][:,0])
            self.line.set_ydata(dist)
        ax.relim(), ax.autoscale(), ax.autoscale_view()
        ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)

    def find_best_snr(self, plot=True, filter_window_length=10):
        res = self._retrieve_results()
        if 'SNR' not in res:
            raise PipelineException("No SNR data yet.")
        SNR = res['SNR']
        if SNR.size > 5:
            filtwindow = min(filter_window_length, SNR.size//2)
            if filtwindow != filter_window_length:
                warnings.warn("Find best SNR: Reducing filter window to half number of samples.")
            filt = savgol_filter(SNR, window_length=filtwindow, polyorder=2)
            lim = np.mean(SNR)+3*np.std(SNR-filt)
        else:
            warnings.warn("Find best SNR: Not applying filter due to small number of samples.")
            filt = SNR
            lim = np.mean(SNR)+3*np.std(SNR)
        bestidx = np.argmax(filt)
        if plot is not None:
            ax = self.ax if plot is True else plot
            ax.plot(res['readout_power'], filt, 'k-')
            ax.axhline(lim, color='gray', linestyle='--', linewidth=1)
            ax.plot([res['readout_power'][bestidx]], [filt[bestidx]], '.', color='r')
        if filt[bestidx] < lim:
            raise PipelineException("SNR signal not clear enough to find peak.")
        return self.params['readout_amps'][bestidx], res['readout_power'][bestidx]


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

    def _figtitle(self, Navg):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        drivepower = opx_amp2pow(self.config['saturation_amp'])
        return (
            f"Time Rabi,   Navg {Navg:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.0f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB,"
            f"  {self.params['drive_read_overlap_cycles']//4}ns overlap"
        )

    def _initialize_liveplot(self, ax):
        durations = self.params['duration_ns']
        self.line, = ax.plot(durations, np.full(len(durations), np.nan))
        ax.set_xlabel("drive duration / ns")
        ax.set_ylabel("arg S")
        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.unwrap(np.angle(res['Z'])))
        ax.relim(), ax.autoscale(), ax.autoscale_view()
        ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)


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
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
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
        self.colorbar = ax.get_figure().colorbar(self.img, ax=ax, orientation='horizontal', shrink=0.9)
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
        self.ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)


class QMTimeRabiChevrons_InnerAvg (QMTimeRabi):
    """Uses short readout pulse and saturation pulse.
    Inherits from QMTimeRabi for check_timing()
    """

    def __init__(self, qmm, config, Nrep, Navg, qubitIFs, max_duration_ns, drive_read_overlap_cycles=0):
        super().__init__(qmm, config, None, None)
        self.params = {
            'Navg': Navg, 'Nrep': Nrep, 'Niter': Nrep,
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
            nrep = qua.declare(int)
            n = qua.declare(int)
            f = qua.declare(int)
            t = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            t_st = qua.declare_stream()
            n_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            with qua.for_(nrep, 0, nrep < self.params['Nrep'], nrep + 1):
                with qua.for_(*from_array(f, qubitIFs)):
                    qua.update_frequency('qubit', f)
                    for l in range(16):
                        qua.save(I, t_st)
                        with qua.for_(n, 0, n < self.params['Navg'], n + 1):
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
                            qua.save(I, t_st)
                            with qua.for_(n, 0, n < self.params['Navg'], n + 1):
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
                qua.save(nrep, n_st)

            with qua.stream_processing():
                n_st.save('iteration')
                t_st.timestamps().buffer(self.params['Nrep'], len(qubitIFs), len(duration_clk)*4+16).save('t')
                I_st.buffer(self.params['Navg']).map(qua.FUNCTIONS.average(0)).buffer(self.params['Nrep'], len(qubitIFs), len(duration_clk)*4+16).save('I')
                Q_st.buffer(self.params['Navg']).map(qua.FUNCTIONS.average(0)).buffer(self.params['Nrep'], len(qubitIFs), len(duration_clk)*4+16).save('Q')

        self.qmprog = prog
        return prog

    def _figtitle(self, Nrep):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        drivepower = opx_amp2pow(self.config['saturation_amp'])
        return (
            f"Rabi Chevrons,   Nrep {Nrep:.2e}   Navg {self.params['Navg']}\n"
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
        self.colorbar = ax.get_figure().colorbar(self.img, ax=ax, orientation='horizontal', shrink=0.9)
        self.colorbar.set_label("arg S")
        ax.set_ylabel("drive IF / MHz")
        ax.set_xlabel("drive duration / ns")
        ax.set_title(self._figtitle(self.params['Nrep']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.img.set_array(np.mean(np.angle(res['Z']), axis=0).T)
        self.img.autoscale()
        self.ax.set_title(self._figtitle((res['iteration'] or 0)), fontsize=8)


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

    @staticmethod
    def cosine(amp, period, a, decay, bkg_slope, bkg_offs):
        """Cosine function with background offset and slope to use for fitting."""
        return -a * np.cos(2*np.pi * amp / period) * np.exp(-amp/decay) + bkg_slope * amp + bkg_offs

    def fit_cosine(
            self, result=None, ax=None, period0=0.05, plotp0=False,
            pltkw={'color': 'k', 'linewidth': 1}, printinfo=True):
        """Fit slanted sine shape to data.

        Assumes first amplitude in drive_amps to be zero (ground state).

        Returns tuple of (optimal, uncertainty), each being an array of
        period, cosine amplitude, background slant, background offset
        """
        res = self._retrieve_results() if result is None else result
        amps = res['drive_amps']
        func = QMPowerRabi.cosine
        signal = np.abs(res['Z'] - res['Z'][0])
        maxmin = np.max(signal)-np.min(signal)
        p0 = self.last_p0 = [
            period0, # period
            maxmin/2.5, # amplitude
            max(amps)/2, # decay
            0, # bkg slant
            np.mean(signal) # bkg offset
        ]
        bounds = (
            [amps[2]-amps[0], 0, 0, -np.inf, np.min(signal)],
            [np.max(amps)*2, maxmin/2, np.inf, np.inf, np.max(signal)])
        print(p0)
        print(bounds)
        popt, pcov = curve_fit(
            func, amps, signal, p0=p0, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        paramnames = ["period", "amplitude", "decay", "slope", "offset"]
        if printinfo:
            res = [ufloat(opt, err) for opt, err in zip(popt, perr)]
            print("  Fit cosine to Power Rabi data")
            for r, name in zip(res, paramnames):
                print(f"    {name:6s} {r}")
        if ax:
            ax.plot(amps, func(amps, *popt), '-', **pltkw)
            if plotp0:
                ax.plot(amps, func(amps, *p0), '--', **pltkw)
        return {
            'signal': signal,
            'p0': p0,
            'parameter_names': paramnames,
            'popt': popt,
            'perr': perr,
            'model': func(amps, *popt)
        }

    def fit_pi_pulse(
            self, result=None, ax=None, period0=0.05, plotp0=False,
            pltkw={'color': 'k', 'linewidth': 1}, printinfo=True):
        """Fit slanted sine shape to data, after selecting first period.

        Assumes first amplitude in drive_amps to be zero (ground state).

        Returns tuple of (optimal, uncertainty), each being an array of
        period, cosine amplitude, background slant, background offset
        """
        res = self._retrieve_results() if result is None else result
        amps = res['drive_amps']
        func = QMPowerRabi.cosine
        signal = fullsignal = np.abs(res['Z'] - res['Z'][0])
        maxmin = np.max(signal)-np.min(signal)

        peaks, props = find_peaks(signal, prominence=maxmin/3)
        if peaks.size and peaks[0] >= 3:
            # cut fit data to 2 * first peak position
            amps = amps[:peaks[0]*2]
            signal = signal[:peaks[0]*2]
            maxmin = np.max(signal)-np.min(signal)
            period0 = amps[peaks[0]]*2

        p0 = self.last_p0 = [
            period0, # period
            maxmin/2.5, # amplitude
            max(amps)/2, # decay
            0, # bkg slant
            np.mean(signal) # bkg offset
        ]
        bounds = (
            [amps[2]-amps[0], 0, 0, -np.inf, np.min(signal)],
            [np.max(amps)*2, maxmin/2, np.inf, np.inf, np.max(signal)])
        print(p0)
        print(bounds)
        popt, pcov = curve_fit(
            func, amps, signal, p0=p0, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        paramnames = ["period", "amplitude", "decay", "slope", "offset"]
        if printinfo:
            uval = [ufloat(opt, err) for opt, err in zip(popt, perr)]
            print("  Fit cosine to Power Rabi data")
            for r, name in zip(uval, paramnames):
                print(f"    {name:6s} {r}")
        if ax:
            ax.plot(amps, func(amps, *popt), '-', **pltkw)
            if plotp0:
                ax.plot(res['drive_amps'][peaks], fullsignal[peaks], '.', **pltkw)
                ax.plot(amps, func(amps, *p0), '--', **pltkw)
        return {
            'drive_amps': amps,
            'signal': signal,
            'p0': p0,
            'parameter_names': paramnames,
            'popt': popt,
            'perr': perr,
            'model': func(amps, *popt)
        }

    def _figtitle(self, Navg):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        return (
            f"Power Rabi,   Navg {Navg:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.0f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.params['duration_ns']:.0f}ns square drive,  {self.config['qubit_output_gain']:+.1f}dB output gain\n"
            f"{self.params['drive_read_overlap_cycles']//4}ns overlap"
        )

    def _initialize_liveplot(self, ax):
        amps = self.params['drive_amps']
        self.line, = ax.plot(amps, np.full(len(amps), np.nan))
        ax.set_xlabel("drive amplitude / V")
        ax.set_ylabel("|S - S0|")
        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.abs(res['Z'] - res['Z'][0]))
        ax.relim(), ax.autoscale(), ax.autoscale_view()
        ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)


class QMPowerRabi_Gaussian (QMPowerRabi):
    """Uses short readout pulse settings for readout and saturation pulse amplitude
    for Gaussian drive pulse amplitude."""

    def __init__(self, qmm, config, Navg, duration_ns, drive_amps, sigma_ns=None, readout_delay_ns=None):
        """

        Parameters
        ----------
        qmm : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.
        Navg : TYPE
            DESCRIPTION.
        duration_ns : TYPE
            Total duration of drive pulse in ns.
            Must be multiple of 4.
        drive_amps : TYPE
            DESCRIPTION.
        sigma_ns : TYPE, optional
            DESCRIPTION. The default is None.
        readout_delay_ns : int, optional
            Must be multiple of 4. If None defaults to duration_ns/2
            rouded up to next multiple of 4.

        Returns
        -------
        None.

        """
        super().__init__(qmm, config, Navg, duration_ns, drive_amps)
        if sigma_ns is None:
            sigma_ns = duration_ns / 4
        if readout_delay_ns is None:
            readout_delay_ns = int(np.ceil(duration_ns / 2 / 4)*4)
        self.params = {
            'Navg': Navg,
            'duration_ns': duration_ns,
            'drive_amps': drive_amps,
            'sigma_ns': sigma_ns,
            'readout_delay_ns': readout_delay_ns}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']
        drive_amp_scale = self.params['drive_amps'] / 0.316
        self.params['drive_power'] = opx_amp2pow(self.params['drive_amps'], self.config['qubit_output_gain'])

        sigma = self.params['sigma_ns']
        duration = tg = self.params['duration_ns']
        assert duration % 8 == 0

        t = np.arange(0, duration, 1)
        pulse = 0.316 * (np.exp(-(t-(tg-1)/2)**2 / (2*sigma**2)) - np.exp(-(tg-1)**2/(8*sigma**2))) / (1-np.exp(-(tg-1)**2/(8*sigma**2)))

        # pad to have at least 16 samples
        wflen = max(16, duration)
        wf = np.zeros(wflen)
        wf[-duration:] = pulse
        tleft = wflen - duration//2 # time until center of pulse
        assert tleft % 4 == 0
        # tleft = duration//2 # time before readout
        # tright = max(16, int(np.ceil(duration/2 / 4)*4)) # time during readout
        # wflen = tleft + tright
        # t = np.arange(wflen) - tleft + sigma + 1
        # wf = np.exp(-(t/sigma)**2 / 2)

        rdelay = self.params['readout_delay_ns']
        assert rdelay % 4 == 0

        # Waveforms are padded to a multiple of 4 samples and a minimum length of 16 samples.
        # Remember: baking modifies the qmconfig but this class instance uses its own deep-copy.
        with baking(self.config['qmconfig'], padding_method='none') as bakedrive:
            bakedrive.add_op('drive_0', 'qubit', [wf, 0*wf])
            bakedrive.play('drive_0', 'qubit')

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
                    qua.wait(12, 'qubit')
                    bakedrive.run(amp_array=[('qubit', a)])
                    qua.wait(12+tleft//4+rdelay//4, 'resonator')
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

    def _figtitle(self, Navg):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        return (
            f"Power Rabi,   Navg {Navg:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.0f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.params['duration_ns']:.0f}ns Gauss drive,  {self.config['qubit_output_gain']:+.1f}dB output gain\n"
        )


class QMRelaxation (QMProgram):
    """Uses short readout pulse and pi pulse amplitude."""

    def __init__(self, qmm, config, Navg, drive_len_ns, max_delay_ns):
        super().__init__(qmm, config)
        self.params = {
            'Navg': Navg,
            'drive_len_ns': drive_len_ns,
            'max_delay_ns': max_delay_ns}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']

        # minimum duration: 4cycles = 16ns
        drivelen = self.params['drive_len_ns']
        maxdelay = self.params['max_delay_ns']
        wflen = max(16, int(np.ceil(drivelen/4)*4)+4)
        self.params['delay_ns'] = np.arange(0, maxdelay, 1)

        assert maxdelay % 4 == 0
        assert maxdelay > 16

        # Remember: baking modifies the qmconfig but this class instance uses its own deep-copy.
        baked_saturation = []
        for l in range(0, 4):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(wflen)
                wf[wflen-drivelen-l:wflen-l] = self.config['pi_amp']
                bake.add_op('drive_%d'%l, 'qubit', [wf, [0]*wflen])
                bake.play('drive_%d'%l, 'qubit')
            baked_saturation.append(bake)

        with qua.program() as prog:
            n = qua.declare(int)
            t4 = qua.declare(int)
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

                # need extra code for t4 < 4 because of wait statement specifics
                for j in range(4):
                    for i in range(4):
                        qua.align()
                        qua.wait(12, 'qubit')
                        baked_saturation[i].run()
                        qua.wait(12+wflen//4 + j, 'resonator')
                        qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(100)+4, 'resonator')

                with qua.for_(t4, 4, t4 < maxdelay//4, t4 + 1):
                    for i in range(4):
                        qua.align()
                        qua.wait(12, 'qubit')
                        baked_saturation[i].run()
                        qua.wait(12+wflen//4, 'resonator')
                        qua.wait(t4, 'resonator')
                        qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(100)+4, 'resonator')

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(maxdelay).average().save('I')
                Q_st.buffer(maxdelay).average().save('Q')

        self.qmprog = prog
        return prog

    def _figtitle(self, Navg):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        drivepower = opx_amp2pow(self.config['pi_amp'])
        return (
            f"Relaxation,  Navg {Navg:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.0f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.params['drive_len_ns']:.0f}ns drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB")

    def _initialize_liveplot(self, ax):
        delays = self.params['delay_ns']
        self.line, = ax.plot(delays, np.full(len(delays), np.nan))
        ax.set_xlabel("measurement delay / ns")
        ax.set_ylabel("| S - S0 |")
        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.abs(res['Z']-res['Z'][0]))
        ax.relim(), ax.autoscale(), ax.autoscale_view()
        ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)

    def check_timing(self, duration_cycles=25000, plot=True):
        """Simulate program and check alignment of drive and readout alignment.
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
        # end of drive pulses, last nonzero sample
        # starts with second drive pulse
        drivestop = np.nonzero(drive)[0][:-1][np.diff(np.nonzero(drive)[0]) > 1][1:]
        # start of readout pulses, first nonzero sample
        # starts with second readout pulse
        readstart = np.nonzero(read)[0][1:][np.diff(np.nonzero(read)[0]) > 1]
        if plot:
            plt.scatter(drivestop, [0]*drivestop.size)
            plt.scatter(readstart, [0]*readstart.size, color='C2')

        l = min(readstart.size, drivestop.size)
        delay = readstart[:l] - drivestop[:l]
        print("Waveform delay, first non-zero readout sample - last non-zero drive sample (ns)")
        print(repr(delay))
        return delay


class QMRamsey (QMProgram):
    """Ramsey sequence: square pi/2 pulse, wait, pi/2 pulse, readout.

    Uses short readout pulse and half of pi_amp pulse amplitude to
    go to the superposition state.

    Since we cannot store an arbitrary number of waveforms we need to bake
    waveforms that can be combined with the qua.wait of multiples of 4ns cycles.
    The second drive pulse needs to be right aligned in a cycle to be followed
    directly by a readout pulse.

    Wait time long enough:
        driveA, then 0 to 3ns wait
        play wait of multiple 4ns cycles
        some ns wait for full 16ns pulse, then driveB right

    For short waits we need to bake the whole pulse sequence into one waveform
    because the qua.wait doesn't have enough resolution:
        driveA, then wait, driveB

    See _make_program implementation for details.
    """

    def __init__(self, qmm, config, Navg, drive_len_ns, max_delay_ns):
        super().__init__(qmm, config)
        self.params = {
            'Navg': Navg,
            'drive_len_ns': drive_len_ns,
            'max_delay_ns': max_delay_ns}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']

        # minimum duration: 4cycles = 16ns
        drivelen = self.params['drive_len_ns']
        maxdelay = self.params['max_delay_ns']
        self.params['delay_ns'] = np.arange(0, maxdelay, 1)

        assert drivelen % 4 == 0 # makes everything so much easier
        assert maxdelay % 4 == 0

        ## Waveforms for long wait case:
        # length of first drive wf including variable wait (1 to 4ns)
        driveAlen = max(16, int(np.ceil((drivelen+4)/4)*4))
        # length of second drive wf, right aligned, multiple of 4ns, minimum 16ns, also initial wait minimum of 16ns
        driveBlen = max(16, int(np.ceil(drivelen/4)*4))
        waitB = driveBlen - drivelen # wait included in driveB
        assert waitB % 4 == 0 # True since drivelen % 4 == 0

        # Delimiter between cases: if wait >= waitB+16 we use two waveforms with >=16ns qua.wait inbetween.

        # wf in case of short wait case:
        # two pi/2 pulses and up to waitB+16ns in between
        shortwflen = max(16, int(np.ceil((2*drivelen+waitB+16)/4)*4))

        # max cycles to wait in qua.wait, (maxdelay-waitB guaranteed multiple of 4ns)
        maxwaitcycles = (maxdelay - waitB) // 4

        pulseamp = self.config['pi_amp'] / 2

        # Remember: baking modifies the qmconfig but this class instance uses its own deep-copy.
        # start pulse driveA, right aligned with 0-3ns wait
        baked_driveA = []
        for l in range(0, 4):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(driveAlen)
                wf[-drivelen-l:] = pulseamp
                if l > 0:
                    wf[-l:] = 0
                bake.add_op('driveA_%d'%l, 'qubit', [wf, [0]*driveAlen])
                bake.play('driveA_%d'%l, 'qubit')
            baked_driveA.append(bake)
        # end pulse driveB, right aligned
        with baking(self.config['qmconfig'], padding_method='none') as baked_driveB:
            wf = np.zeros(driveBlen)
            wf[-drivelen:] = pulseamp
            baked_driveB.add_op('driveB', 'qubit', [wf, [0]*driveBlen])
            baked_driveB.play('driveB', 'qubit')

        ## Waveforms for short wait case, all pulses baked into one wf:
        baked_driveshort = []
        for l in range(0, waitB+16):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(shortwflen)
                wf[-2*drivelen-l:] = 1
                wf[-drivelen-l:] = 0
                wf[-drivelen:] = 1
                bake.add_op('driveshort_%d'%l, 'qubit', [wf*pulseamp, [0]*shortwflen])
                bake.play('driveshort_%d'%l, 'qubit')
            baked_driveshort.append(bake)

        with qua.program() as prog:
            n = qua.declare(int)
            t4 = qua.declare(int)
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

                # short case, not including wait==waitB+16
                for j in range(waitB+16):
                    qua.align()
                    qua.wait(12, 'qubit')
                    baked_driveshort[j].run()
                    qua.wait(12+shortwflen//4, 'resonator')
                    qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                        qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                        qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                    qua.save(I, I_st)
                    qua.save(Q, Q_st)
                    qua.wait(self.config['cooldown_clk'], 'resonator')
                    qua.wait(rand.rand_int(50)+4, 'resonator')

                with qua.for_(t4, 4, t4 < maxwaitcycles, t4 + 1):
                    for i in range(4):
                        qua.align()
                        # qubit pulses
                        qua.wait(12, 'qubit')
                        baked_driveA[i].run()
                        qua.wait(t4, 'qubit')
                        baked_driveB.run()
                        # readout
                        qua.wait(12+driveAlen//4+driveBlen//4, 'resonator')
                        qua.wait(t4, 'resonator')
                        qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator')

            with qua.stream_processing():
                n_st.save('iteration')
                I_st.buffer(maxdelay).average().save('I')
                Q_st.buffer(maxdelay).average().save('Q')

        self.qmprog = prog
        return prog

    def _figtitle(self, Navg):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        drivepower = opx_amp2pow(self.config['pi_amp']/2)
        return (
            f"Ramsey,  Navg {Navg:.2e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.0f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.params['drive_len_ns']:.0f}ns drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB")

    def _initialize_liveplot(self, ax):
        delays = self.params['delay_ns']
        self.line, = ax.plot(delays, np.full(len(delays), np.nan))
        ax.set_xlabel("pulse delay / ns")
        ax.set_ylabel("arg S")
        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        self.line.set_ydata(np.unwrap(np.angle(res['Z'])))
        ax.relim(), ax.autoscale(), ax.autoscale_view()
        ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)


class QMRamseyRepeat (QMProgram):
    """Like QMRamsey but repeated. Saves average every Navg samples.

    Uses short readout pulse. Uses square pulse with half of
    pi_amp pulse amplitude to go to the superposition state.
    """

    def __init__(self, qmm, config, Nrep, Navg, drive_len_ns, max_delay_ns):
        super().__init__(qmm, config)
        self.params = {
            'Navg': Navg, 'Nrep': Nrep, 'Niter': Navg*Nrep,
            'drive_len_ns': drive_len_ns,
            'max_delay_ns': max_delay_ns}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']

        # minimum duration: 4cycles = 16ns
        drivelen = self.params['drive_len_ns']
        maxdelay = self.params['max_delay_ns']
        self.params['delay_ns'] = np.arange(0, maxdelay, 1)

        assert drivelen % 4 == 0 # makes everything so much easier
        assert maxdelay % 4 == 0

        ## Waveforms for long wait case:
        # length of first drive wf including variable wait (1 to 4ns)
        driveAlen = max(16, int(np.ceil((drivelen+4)/4)*4))
        # length of second drive wf, right aligned, multiple of 4ns, minimum 16ns, also initial wait minimum of 16ns
        driveBlen = max(16, int(np.ceil(drivelen/4)*4))
        waitB = driveBlen - drivelen # wait included in driveB
        assert waitB % 4 == 0 # True since drivelen % 4 == 0

        # Delimiter between cases: if wait >= waitB+16 we use two waveforms with >=16ns qua.wait inbetween.

        # wf in case of short wait case:
        # two pi/2 pulses and up to waitB+16ns in between
        shortwflen = max(16, int(np.ceil((2*drivelen+waitB+16)/4)*4))

        # max cycles to wait in qua.wait, (maxdelay-waitB guaranteed multiple of 4ns)
        maxwaitcycles = (maxdelay - waitB) // 4

        pulseamp = self.config['pi_amp'] / 2

        # Remember: baking modifies the qmconfig but this class instance uses its own deep-copy.
        # start pulse driveA, right aligned with 0-3ns wait
        baked_driveA = []
        for l in range(0, 4):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(driveAlen)
                wf[-drivelen-l:] = pulseamp
                if l > 0:
                    wf[-l:] = 0
                bake.add_op('driveA_%d'%l, 'qubit', [wf, [0]*driveAlen])
                bake.play('driveA_%d'%l, 'qubit')
            baked_driveA.append(bake)
        # end pulse driveB, right aligned
        with baking(self.config['qmconfig'], padding_method='none') as baked_driveB:
            wf = np.zeros(driveBlen)
            wf[-drivelen:] = pulseamp
            baked_driveB.add_op('driveB', 'qubit', [wf, [0]*driveBlen])
            baked_driveB.play('driveB', 'qubit')

        ## Waveforms for short wait case, all pulses baked into one wf:
        baked_driveshort = []
        for l in range(0, waitB+16):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(shortwflen)
                wf[-2*drivelen-l:] = 1
                wf[-drivelen-l:] = 0
                wf[-drivelen:] = 1
                bake.add_op('driveshort_%d'%l, 'qubit', [wf*pulseamp, [0]*shortwflen])
                bake.play('driveshort_%d'%l, 'qubit')
            baked_driveshort.append(bake)

        with qua.program() as prog:
            m = qua.declare(int)
            n = qua.declare(int)
            iterations = qua.declare(int)
            t4 = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            t_st = qua.declare_stream() # for timestamps
            n_st = qua.declare_stream()
            I_st = qua.declare_stream()
            Q_st = qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            qua.update_frequency('qubit', self.config['qubitIF'])
            with qua.for_(m, 0, m < self.params['Nrep'], m + 1):
                qua.save(m, t_st)
                with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                    qua.assign(iterations, m*self.params['Navg']+n)
                    qua.save(iterations, n_st)

                    # short case, not including wait==waitB+16
                    for j in range(waitB+16):
                        qua.align()
                        qua.wait(12, 'qubit')
                        baked_driveshort[j].run()
                        qua.wait(12+shortwflen//4, 'resonator')
                        qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, I_st)
                        qua.save(Q, Q_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator')

                    with qua.for_(t4, 4, t4 < maxwaitcycles, t4 + 1):
                        for i in range(4):
                            qua.align()
                            # qubit pulses
                            qua.wait(12, 'qubit')
                            baked_driveA[i].run()
                            qua.wait(t4, 'qubit')
                            baked_driveB.run()
                            # readout
                            qua.wait(12+driveAlen//4+driveBlen//4, 'resonator')
                            qua.wait(t4, 'resonator')
                            qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                            qua.save(I, I_st)
                            qua.save(Q, Q_st)
                            qua.wait(self.config['cooldown_clk'], 'resonator')
                            qua.wait(rand.rand_int(50)+4, 'resonator')

            with qua.stream_processing():
                n_st.save('iteration')
                t_st.with_timestamps().save_all('t')
                I_st.buffer(self.params['Navg'], maxdelay).map(qua.FUNCTIONS.average(0)).save_all('I')
                Q_st.buffer(self.params['Navg'], maxdelay).map(qua.FUNCTIONS.average(0)).save_all('Q')

        self.qmprog = prog
        return prog

    def _figtitle(self, Niter):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        drivepower = opx_amp2pow(self.config['pi_amp']/2)
        return (
            f"Ramsey repetitions,  Niter {Niter:.2e}, Navg {self.params['Navg']:.1e}, Nrep {self.params['Nrep']:.1e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz{self.config['qubitIF']/1e6:+.0f}MHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.params['drive_len_ns']:.0f}ns drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB")

    def _initialize_liveplot(self, ax):
        delays = self.params['delay_ns']
        self.lines = [ax.plot(delays, np.full(len(delays), np.nan))[0] for i in range(self.params['Nrep'])]
        ax.set_xlabel("pulse delay / ns")
        ax.set_ylabel("arg S")
        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        for i in range(res['Z'].shape[0]):
            self.lines[i].set_ydata(np.unwrap(np.angle(res['Z'][i])))
        ax.relim(), ax.autoscale(), ax.autoscale_view()
        ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)


class QMRamseyChevronRepeat (QMProgram):
    """QMRamseyRepeat with variable qubit drive frequency.

    Uses short readout pulse and half of pi_amp pulse amplitude to
    go to the superposition state.
    """

    def __init__(self, qmm, config, Nrep, Navg, qubitIFs, drive_len_ns, max_delay_ns):
        super().__init__(qmm, config)
        self.params = {
            'Navg': Navg, 'Nrep': Nrep, 'Niter': Navg*Nrep,
            'qubitIFs': qubitIFs,
            'drive_len_ns': drive_len_ns,
            'max_delay_ns': max_delay_ns}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']
        qubitIFs = self.params['qubitIFs']

        # minimum duration: 4cycles = 16ns
        drivelen = self.params['drive_len_ns']
        maxdelay = self.params['max_delay_ns']
        self.params['delay_ns'] = np.arange(0, maxdelay, 1)

        assert drivelen % 4 == 0 # makes everything so much easier
        assert maxdelay % 4 == 0

        ## Waveforms for long wait case:
        # length of first drive wf including variable wait (1 to 4ns)
        driveAlen = max(16, int(np.ceil((drivelen+4)/4)*4))
        # length of second drive wf, right aligned, multiple of 4ns, minimum 16ns, also initial wait minimum of 16ns
        driveBlen = max(16, int(np.ceil(drivelen/4)*4))
        waitB = driveBlen - drivelen # wait included in driveB
        assert waitB % 4 == 0 # True since drivelen % 4 == 0

        # Delimiter between cases: if wait >= waitB+16 we use two waveforms with >=16ns qua.wait inbetween.

        # wf in case of short wait case:
        # two pi/2 pulses and up to waitB+16ns in between
        shortwflen = max(16, int(np.ceil((2*drivelen+waitB+16)/4)*4))

        # max cycles to wait in qua.wait, (maxdelay-waitB guaranteed multiple of 4ns)
        maxwaitcycles = (maxdelay - waitB) // 4

        pulseamp = self.config['pi_amp'] / 2

        # Remember: baking modifies the qmconfig but this class instance uses its own deep-copy.
        # start pulse driveA, right aligned with 0-3ns wait
        baked_driveA = []
        for l in range(0, 4):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(driveAlen)
                wf[-drivelen-l:] = pulseamp
                if l > 0:
                    wf[-l:] = 0
                bake.add_op('driveA_%d'%l, 'qubit', [wf, [0]*driveAlen])
                bake.play('driveA_%d'%l, 'qubit')
            baked_driveA.append(bake)
        # end pulse driveB, right aligned
        with baking(self.config['qmconfig'], padding_method='none') as baked_driveB:
            wf = np.zeros(driveBlen)
            wf[-drivelen:] = pulseamp
            baked_driveB.add_op('driveB', 'qubit', [wf, [0]*driveBlen])
            baked_driveB.play('driveB', 'qubit')

        ## Waveforms for short wait case, all pulses baked into one wf:
        baked_driveshort = []
        for l in range(0, waitB+16):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(shortwflen)
                wf[-2*drivelen-l:] = 1
                wf[-drivelen-l:] = 0
                wf[-drivelen:] = 1
                bake.add_op('driveshort_%d'%l, 'qubit', [wf*pulseamp, [0]*shortwflen])
                bake.play('driveshort_%d'%l, 'qubit')
            baked_driveshort.append(bake)

        with qua.program() as prog:
            m = qua.declare(int)
            n = qua.declare(int)
            iterations = qua.declare(int)
            f = qua.declare(int)
            t4 = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            t_st = qua.declare_stream() # for timestamps
            n_st = qua.declare_stream()
            I_st, Q_st = qua.declare_stream(), qua.declare_stream()
            Ig_st, Qg_st = qua.declare_stream(), qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            qua.update_frequency('qubit', self.config['qubitIF'])
            with qua.for_(m, 0, m < self.params['Nrep'], m + 1):
                qua.save(m, t_st)
                with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                    qua.assign(iterations, m*self.params['Navg']+n)
                    qua.save(iterations, n_st)

                    with qua.for_(*from_array(f, qubitIFs)):
                        qua.update_frequency('qubit', f)

                        # Ground state reference, no drive
                        qua.align()
                        qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, Ig_st)
                        qua.save(Q, Qg_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator')

                        # short case, not including wait==waitB+16
                        for j in range(waitB+16):
                            qua.align()
                            qua.wait(12, 'qubit')
                            baked_driveshort[j].run()
                            qua.wait(12+shortwflen//4, 'resonator')
                            qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                            qua.save(I, I_st)
                            qua.save(Q, Q_st)
                            qua.wait(self.config['cooldown_clk'], 'resonator')
                            qua.wait(rand.rand_int(50)+4, 'resonator')

                        with qua.for_(t4, 4, t4 < maxwaitcycles, t4 + 1):
                            for i in range(4):
                                qua.align()
                                # qubit pulses
                                qua.wait(12, 'qubit')
                                baked_driveA[i].run()
                                qua.wait(t4, 'qubit')
                                baked_driveB.run()
                                # readout
                                qua.wait(12+driveAlen//4+driveBlen//4, 'resonator')
                                qua.wait(t4, 'resonator')
                                qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                                qua.save(I, I_st)
                                qua.save(Q, Q_st)
                                qua.wait(self.config['cooldown_clk'], 'resonator')
                                qua.wait(rand.rand_int(50)+4, 'resonator')

            with qua.stream_processing():
                n_st.save('iteration')
                t_st.with_timestamps().save_all('t')
                I_st.buffer(self.params['Navg'], len(qubitIFs), maxdelay).map(qua.FUNCTIONS.average(0)).save_all('I')
                Q_st.buffer(self.params['Navg'], len(qubitIFs), maxdelay).map(qua.FUNCTIONS.average(0)).save_all('Q')
                Ig_st.buffer(self.params['Navg'], len(qubitIFs)).map(qua.FUNCTIONS.average(0)).save_all('Ig')
                Qg_st.buffer(self.params['Navg'], len(qubitIFs)).map(qua.FUNCTIONS.average(0)).save_all('Qg')

        self.qmprog = prog
        return prog

    def _retrieve_results(self, resulthandles=None):
        """Also merges Ig and Qg into Zg."""
        res = super()._retrieve_results(resulthandles)
        if res['Ig'] is not None and res['Qg'] is not None:
            try:
                res['Zg'] = res['Ig'] + 1j * res['Qg']
            except ValueError:
                res['Zg'] = None
        return res

    def _figtitle(self, Niter):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        drivepower = opx_amp2pow(self.config['pi_amp']/2)
        return (
            f"Ramsey chevrons, repetitions\nNiter {Niter:.2e}, Navg {self.params['Navg']:.1e}, Nrep {self.params['Nrep']:.1e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.params['drive_len_ns']:.0f}ns square drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB")

    def _initialize_liveplot(self, ax):
        delays = self.params['delay_ns']
        freqs = self.params['qubitIFs']
        xx, yy = np.meshgrid(delays, freqs/1e6, indexing='ij')
        self.img = ax.pcolormesh(xx, yy, np.full(
            (len(delays), len(freqs)), np.nan), shading='nearest')
        self.colorbar = ax.get_figure().colorbar(self.img, ax=ax, orientation='horizontal', shrink=0.9)
        self.colorbar.set_label("arg S")
        ax.set_ylabel("drive IF / MHz")
        ax.set_xlabel("delay / ns")
        ax.set_title(self._figtitle(self.params['Navg']), fontsize=8)
        self.ax = ax

        self.ax2 = ax.twinx()
        x = np.linspace(0, max(delays), self.params['Nrep'])
        self.line, = self.ax2.plot(x, np.full(self.params['Nrep'], np.nan), color='k', linewidth=2, zorder=100)

    def _update_liveplot(self, ax, resulthandles):
        res = self._retrieve_results(resulthandles)
        if res['Z'] is None:
            return
        argS = np.unwrap(np.unwrap(np.angle(np.mean(res['Z'], axis=0)), axis=0))
        self.img.set_array(argS.T)
        self.img.autoscale()
        ax.set_title(self._figtitle((res['iteration'] or 0)+1), fontsize=8)

        meanS = np.full(self.params['Nrep'], np.nan)
        meanS[:res['Z'].shape[0]] = np.unwrap(np.angle(np.mean(res['Z'], axis=(1,2))))
        self.line.set_ydata(meanS)
        self.ax2.relim(), self.ax2.autoscale(), self.ax2.autoscale_view()

    def clear_liveplot(self):
        if hasattr(self, 'ax2'):
            self.ax2.remove()
        super().clear_liveplot()


class QMRamseyChevronRepeat_Gaussian (QMRamseyChevronRepeat):
    r"""Ramsey sequence varying drive frequency, using Gaussian drive pulses.

    Result compatible with QMRamseyChevronRepeat (same fields and shape).

    Uses short readout pulse for readout.
    Uses and half of pi_amp pulse amplitude as amplitude of Gaussian pulse.
    Amplitude is the maximum value of the envelope.

    The Gaussian pulse with total duration $t_g$ and width $\sigma$ is

    $$A \frac{e^{-(t-(t_g-1)/2)^2 / (2 σ^2)} - e^{-(t_g-1)^2/(8 σ^2)}}{1 - e^{-(t_g-1)^2/(8 σ^2)}}$$

    where $t=[0,1,2,...t_g)$ in ns. The substraction of $e^{-(t_g-1)^2/(8 σ^2)}$
    makes sure that the signal is zero at beginning and and of drive
    (i.e. the first and last sample of the drive pulse).
    The offset of 1ns on $t_g$ centers the Gaussian in the interval.
    Both of these corrections reduce leakage in frequency space.

    drive_len_ns ($t_g$) is the length of the drive pulse, and needs to be
    a multiple of 4ns for alignment with QM clock cycles. Since it is even, the
    Peak of the Gaussian lies between samples. For alignment purposes the
    center is taken to be at $t_g/2$.

    The width of the Gaussian, sigma_ns, can be specified independently (should
    be smaller than drive_len_ns), and if None defaults to
    sigma_ns=drive_len_ns/4.

    Where the pulses overlap, their amplitudes are added.

    The delay between pulses is given by the Gaussian center (i.e. at
    zero delay, they are at the same time and we get just twice the same
    Gaussian pulse). Note that this definition is different from the Ramsey
    sequence with square pulses where the delay is strictly between the pulses.

    Demodulation data has shape: (Nrep, qubitIFs, delay_ns).
    """

    def __init__(
            self, qmm, config, Nrep, Navg, qubitIFs,
            max_delay_ns, drive_len_ns, sigma_ns=None, readout_delay_ns=None):
        """
        Parameters
        ----------
        qmm : qm.QuantumMachineManager
            Used to open new QuantumMachine
        config : dict
        Nrep : int
            Number of repetitions.
        Navg : int
            Number of averages per repetition.
        qubitIFs : numpy.ndarray
            Drive intermediate frequencies.
        max_delay_ns : int
            Maximum pulse delay. Will run protocol for delays from 0ns to
            max_delay_ns. Must be multiple of 4.
        drive_len_ns : int
            Total length of Gaussian pulse in ns.
            Must be multiple of 8ns.
        sigma_ns : float, optional
            Width of Gaussian. If None, defaults to drive_len_ns / 4.
        readout_delay_ns : int, optional
            Delay of readout pulse after center of last pulse.
            Must be multiple of 4ns.
            If None, defaults to drive_len_ns/2 rounded up to next multiple of 4.
        """
        super().__init__(qmm, config, Nrep, Navg, qubitIFs, drive_len_ns, max_delay_ns)
        if sigma_ns is None:
            sigma_ns = drive_len_ns / 4
        if readout_delay_ns is None:
            readout_delay_ns = int(np.ceil(drive_len_ns / 2 / 4)*4)
        self.params = {
            'Navg': Navg, 'Nrep': Nrep, 'Niter': Navg*Nrep,
            'qubitIFs': qubitIFs,
            'drive_len_ns': drive_len_ns,
            'sigma_ns': sigma_ns,
            'max_delay_ns': max_delay_ns,
            'readout_delay_ns': readout_delay_ns}

    def _make_program(self):
        read_amp_scale = self.config['short_readout_amp'] / \
            self.config['qmconfig']['waveforms']['short_readout_wf']['sample']
        qubitIFs = self.params['qubitIFs']

        readoutdelay = self.params['readout_delay_ns']
        assert readoutdelay % 4 == 0

        maxdelay = self.params['max_delay_ns']
        assert maxdelay % 4 == 0
        self.params['delay_ns'] = np.arange(0, maxdelay, 1)

        pulseamp = self.config['pi_amp'] / 2
        sigma = self.params['sigma_ns']
        drivelen = tg = self.params['drive_len_ns']
        assert drivelen % 8 == 0

        t = np.arange(0, drivelen, 1)
        pulse = pulseamp * (np.exp(-(t-(tg-1)/2)**2 / (2*sigma**2)) - np.exp(-(tg-1)**2/(8*sigma**2))) / (1-np.exp(-(tg-1)**2/(8*sigma**2)))

        ## Waveforms for long wait case:
        # length of first drive wf, right aligned,
        # including variable wait (1 to 4ns),
        # min wf length is 16 samples and multiple of 4ns
        driveAlen = max(16, drivelen+4)
        # length of second drive wf, right aligned
        driveBlen = max(16, drivelen)
        # wait included in driveA after Gaussian center
        waitA = drivelen//2
        # wait included in driveB before Gaussian center
        waitB = driveBlen - drivelen//2
        assert (waitA+waitB) % 4 == 0 # for maxwaitcycles later
        # Number of cycles to wait before readout,
        # this is the reason drive_len_ns needs to be multiple of 8
        longwfminreadoutcycles = (driveAlen+driveBlen-drivelen//2+readoutdelay)//4

        # Remember: baking modifies the qmconfig but every instance uses its own deep-copy.
        # start pulse driveA, right aligned with 0-3ns wait
        baked_driveA = []
        for l in range(0, 4):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(driveAlen)
                wf[wf.size-drivelen-l : wf.size-l] = pulse
                bake.add_op('driveA_%d'%l, 'qubit', [wf, [0]*driveAlen])
                bake.play('driveA_%d'%l, 'qubit')
            baked_driveA.append(bake)
        # end pulse driveB, right aligned
        with baking(self.config['qmconfig'], padding_method='none') as baked_driveB:
            wf = np.zeros(driveBlen)
            wf[-drivelen:] = pulse
            baked_driveB.add_op('driveB', 'qubit', [wf, [0]*driveBlen])
            baked_driveB.play('driveB', 'qubit')

        # Limit between cases: if wait >= waitA+waitB+16 we use two waveforms with >=16ns qua.wait inbetween.

        ## Waveform in case of short wait case:
        # two pi/2 pulses and up to waitA+waitB+16ns in between
        shortwflen = max(16, 2*drivelen+waitA+waitB+16)
        # Second pulse center at
        shortwfend = shortwflen - drivelen//2
        # max cycles to wait in qua.wait
        maxwaitcycles = (maxdelay-waitB-waitA) // 4

        ## Waveforms for short wait case, all pulses baked into one wf.
        baked_driveshort = []
        for l in range(0, waitA+waitB+16):
            with baking(self.config['qmconfig'], padding_method='none') as bake:
                wf = np.zeros(shortwflen)
                wf[shortwflen-drivelen-l:shortwflen-l] += pulse
                wf[-drivelen:] += pulse
                bake.add_op('driveshort_%d'%l, 'qubit', [wf, [0]*shortwflen])
                bake.play('driveshort_%d'%l, 'qubit')
            baked_driveshort.append(bake)

        print('sigma', sigma)
        print('drivelen', drivelen)
        print('driveAlen', driveAlen)
        print('driveBlen', driveBlen)
        print('waitA', waitA)
        print('waitB', waitB)
        print('longwfminreadoutcycles', longwfminreadoutcycles)
        print('shortwflen', shortwflen)
        print('shortwfend', shortwfend)
        print('maxwaitcycles', maxwaitcycles)

        with qua.program() as prog:
            m = qua.declare(int)
            n = qua.declare(int)
            iterations = qua.declare(int)
            f = qua.declare(int)
            t4 = qua.declare(int)
            I = qua.declare(qua.fixed)
            Q = qua.declare(qua.fixed)
            t_st = qua.declare_stream() # for timestamps
            n_st = qua.declare_stream()
            I_st, Q_st = qua.declare_stream(), qua.declare_stream()
            Ig_st, Qg_st = qua.declare_stream(), qua.declare_stream()
            rand = qua.lib.Random()

            qua.update_frequency('resonator', self.config['resonatorIF'])
            qua.update_frequency('qubit', self.config['qubitIF'])
            with qua.for_(m, 0, m < self.params['Nrep'], m + 1):
                qua.save(m, t_st)
                with qua.for_(n, 0, n < self.params['Navg'], n + 1):
                    qua.assign(iterations, m*self.params['Navg']+n)
                    qua.save(iterations, n_st)

                    with qua.for_(*from_array(f, qubitIFs)):
                        qua.update_frequency('qubit', f)

                        # Ground state reference, no drive
                        qua.align()
                        qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                            qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                            qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                        qua.save(I, Ig_st)
                        qua.save(Q, Qg_st)
                        qua.wait(self.config['cooldown_clk'], 'resonator')
                        qua.wait(rand.rand_int(50)+4, 'resonator')

                        # short wait case
                        for j in range(waitA+waitB+16):
                            qua.align()
                            qua.wait(12, 'qubit')
                            baked_driveshort[j].run()
                            qua.wait(12+shortwfend//4+readoutdelay//4, 'resonator')
                            qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                                qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                                qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                            qua.save(I, I_st)
                            qua.save(Q, Q_st)
                            qua.wait(self.config['cooldown_clk'], 'resonator')
                            qua.wait(rand.rand_int(50)+4, 'resonator')

                        with qua.for_(t4, 4, t4 < maxwaitcycles, t4 + 1):
                            for i in range(4):
                                qua.align()
                                # qubit pulses
                                qua.wait(12, 'qubit')
                                baked_driveA[i].run()
                                qua.wait(t4, 'qubit')
                                baked_driveB.run()
                                # readout
                                qua.wait(12+longwfminreadoutcycles, 'resonator')
                                qua.wait(t4, 'resonator')
                                qua.measure('short_readout'*qua.amp(read_amp_scale), 'resonator', None,
                                    qua.dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                                    qua.dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))
                                qua.save(I, I_st)
                                qua.save(Q, Q_st)
                                qua.wait(self.config['cooldown_clk'], 'resonator')
                                qua.wait(rand.rand_int(50)+4, 'resonator')

            with qua.stream_processing():
                n_st.save('iteration')
                t_st.with_timestamps().save_all('t')
                I_st.buffer(self.params['Navg'], len(qubitIFs), maxdelay).map(qua.FUNCTIONS.average(0)).save_all('I')
                Q_st.buffer(self.params['Navg'], len(qubitIFs), maxdelay).map(qua.FUNCTIONS.average(0)).save_all('Q')
                Ig_st.buffer(self.params['Navg'], len(qubitIFs)).map(qua.FUNCTIONS.average(0)).save_all('Ig')
                Qg_st.buffer(self.params['Navg'], len(qubitIFs)).map(qua.FUNCTIONS.average(0)).save_all('Qg')

        self.qmprog = prog
        return prog

    def _figtitle(self, Niter):
        readoutpower = opx_amp2pow(self.config['short_readout_amp'])
        drivepower = opx_amp2pow(self.config['pi_amp']/2)
        return (
            f"Ramsey chevrons, repetitions\nNiter {Niter:.2e}, Navg {self.params['Navg']:.1e}, Nrep {self.params['Nrep']:.1e}\n"
            f"resonator {self.config['resonatorLO']/1e9:.3f}GHz{self.config['resonatorIF']/1e6:+.3f}MHz\n"
            f"qubit {self.config['qubitLO']/1e9:.3f}GHz\n"
            f"{self.config['short_readout_len']:.0f}ns readout at {readoutpower:.1f}dBm{self.config['resonator_output_gain']:+.1f}dB\n"
            f"{self.params['drive_len_ns']:.0f}ns Gauss drive at {drivepower:.1f}dBm{self.config['qubit_output_gain']:+.1f}dB")

    def check_timing(self, duration_cycles=20000, plot=True):
        """Simulate program and check alignment of drive and readout alignment.
        Raises PipelineException if the alignment is not constant for all pulses.
        Returns the alignment timing in ns.

        Analog output port numbers are hardcoded! TODO
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
        # Time of first non-zero drive samples
        drivestart = np.nonzero(drive)[0][1:][np.diff(np.nonzero(drive)[0]) > 1]
        # Time of last non-zero drive samples
        drivestop = np.nonzero(drive)[0][:-1][np.diff(np.nonzero(drive)[0]) > 1]
        # start of readout pulses, excluding first one
        readstart = np.nonzero(read)[0][1:][np.diff(np.nonzero(read)[0]) > 1]
        if plot:
            plt.scatter(drivestart, [0]*drivestart.size)
            plt.scatter(drivestop, [0]*drivestop.size)
            plt.scatter(readstart, [0]*readstart.size)

        # Distance between two drive pulses
        l = min(drivestart.size, drivestop.size)
        dlen = drivestop[1:l]-drivestart[:l-1]+1
        print("Drive pulse length (non-zero samples), excluding first pulse")
        print(repr(dlen[dlen<100]))

        # drive pulses closer than 100ns to each other
        # assert each shot separated by at least 100ns
        ddist = np.diff(drivestart)
        print("Drive pulse distance (first non-zero to first non-zero sample")
        print(repr(ddist[ddist<100]))
        assert np.all(np.diff(ddist[ddist<100]) == 1)

        # readout delay
        dend = drivestop[:-1][np.diff(drivestop)>100]
        l = min(dend.size, readstart.size)
        print("Readout start - end of last drive pulse")
        print(repr(readstart[:l-1]-dend[1:l]))

        return drivestart, drivestop, readstart

        # l = min(readstart.size, drivestop.size)
        # overlap = drivestop[:l] - readstart[:l]
        # print("Waveform overlap:", overlap, "ns")
        # if not np.all(overlap == overlap[0]):
        #     raise PipelineException("Drive/readout waveform alignment not constant.")
        # return overlap

# %%

if __name__ == '__main__':
    import importlib
    import configuration_pipeline_sim as config
    # import configuration as config
    importlib.reload(config)

    qmm = qminit.connect()

    # QMMixerCalibration(qmm, config).run()
    # p = QMTimeOfFlight(qmm, config, Navg=100)
    # results = p.run()
    # print(results.keys())

    # config.readout_amp = 0.0316
    # p = QMResonatorSpec(qmm, config, Navg=500,
    #                     resonatorIFs=np.arange(202e6, 212e6, 0.05e6))
    # results = p.run(plot=True)

    # p = QMNoiseSpectrum(qmm, config, Nsamples=100000, wait_ns=16)
    # results = p.run(plot=True)

    # config.resonator_output_gain = 10
    # p = QMResonatorSpec_P2(
    #     qmm, config, Navg=100,
    #     resonatorIFs=np.arange(203e6, 209e6, 0.1e6),
    #     readout_amps=np.logspace(np.log10(0.000316), np.log10(0.0316), 21))
    # results = p.run(plot=True)
    # config.resonator_output_gain = -20

    # config.saturation_amp = 0.1
    # p = QMQubitSpec(qmm, config, Navg=1000,
    #                 qubitIFs=np.arange(50e6, 450e6, 5e6))
    # results = p.run(plot=True)

    # p = QMQubitSpecThreeTone(qmm, config, Navg=100, third_amp=0.12, thirdIFs=np.linspace(-100e6, -200e6, 11))
    # p.simulate(20000, plot=True)

    # p = QMQubitSpec_P2(
    #     qmm, config, Navg=500, qubitIFs=np.arange(50e6, 450e6, 5e6),
    #     drive_amps=np.logspace(np.log10(0.01), np.log10(0.316), 11))
    # results = p.run(plot=True)
    
    # config.qubitIF = 0
    # p = QMPowerRabi_Gaussian(qmm, config, Navg=1e6, duration_ns=16, sigma_ns=4, drive_amps=np.linspace(0, 0.1, 5))
    # p.simulate(20000, plot=True)

    config.qubitIF = 0
    p = QMRelaxation(qmm, config, Navg=100, drive_len_ns=8, max_delay_ns=52)
    p.check_timing(20000, plot=True)

    # p = QMRamsey(qmm, config, Navg=100, drive_len_ns=8, max_delay_ns=52)
    # p.simulate(300000, plot=True)

    # p = QMRamseyRepeat(qmm, config, Nrep=10, Navg=1000, drive_len_ns=8, max_delay_ns=52)
    # results = p.run(plot=True)
    # p.simulate(30000, plot=True)

    # p = QMRamseyChevronRepeat(qmm, config, qubitIFs=np.linspace(-200e6, -50e6, 11), Nrep=10, Navg=1000, drive_len_ns=8, max_delay_ns=52)
    # results = p.run(plot=True)

    # p = QMRamseyChevronRepeat_Gaussian(
    #     qmm, config, qubitIFs=np.linspace(0, 10e6, 11), Nrep=10, Navg=1000,
    #     drive_len_ns=32, sigma_ns=4, max_delay_ns=100, readout_delay_ns=4)
    # #p.simulate(30000, plot=True)
    # dstart, dstop, rstart = p.check_timing()
    # # results = p.run(plot=True)

#%%

# if __name__ == '__main__':
#     fig, ax = plt.subplots()
#     p = QMNoiseSpectrum(qmm, config, Nsamples=50000, wait_ns=16)
#     p._initialize_liveplot(ax)
#     for i in range(1000):
#         results = p.run(plot=False)
#         p._update_liveplot(ax, p.last_job.result_handles)
#         ax.set_xlim(-110, 110)
#         mpl_pause(1)
