"""
Fit functions for time domain measurements. Runs on result dictionaries from qmtools programs.

Contains functions for fitting data from qmtools programs.
Names follow the convention 'fit_QMTOOLSNAME()' with suffixes if there are
multiple ways to treat the data. These function return optimal parameters,
the modeled data and the signal used in the fit.

Furthermore there are function to extract coherence information:

  - fit_T1 for a relaxation measurement
  - fit_T2 for any Ramsey measurement
  - fit_T2E for a Hahn echo measurement

These return a dict with the keys

  - type: string, one of: T1, T2*, T2E
  - T: float (ns)
  - Terr: float (ns)
  - drive_len: int (ns) drive pulse duration (see protocol description)
  - fdrive: float, protocol frequency if only one, or qubit frequency assumed from spectroscopy.
  - fdriveLO: float, qubit drive LO frequency
  - pi_amp: float (V), pi amplitude calibrated there (based on config, independent from actual used pulse in that protocol)

"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from uncertainties import ufloat


# TODO fit time_rabi |Z-Z(t0)|
# TODO fit time_rabi_chevron |Z-<Z(t0)>|
# TODO fit power_rabi
# TODO fit ramsey argZ
# TODO fit ramsey chevron argZ (if Zq not available)


def fit_relaxation(res, print_info=True, silent_exceptions=True):
    """For 'relaxation:...' measurement, fit exponential decay.

    We don't have the ground state signal, because it is not guaranteed that the time series is much longer than T1.
    Instead we use the decay away from the excited state (or wherever we are) at t=0.
    Unfortunately may not have very good SNR on first point.

    Rejects the result and returns nan if

    - there is a RuntimeError during curve_fit, i.e. no convergence
    - any error bar is not finite or
    - T1 longer than 3 times the measurement duration

    Returns
    -------
    popt : array of 2 values
        (T1, amplitude) best fit values or NaNs.
    perr : array of 2 values
    model : 1D array
        Model result
    signal : 1D array
        Signal used for fitting
    ts : 1D array
        Pulse delay in ns corresponding to signal.
    """
    ts = res['delay_ns']
    signal = np.abs(res['Z'] - res['Z'][0])

    def expfit(t, t1, amp):
        return amp * (1-np.exp(-t / t1))

    p0 = [max(ts)/2, signal[-1]]
    try:
        popt, pcov = curve_fit(expfit, ts, signal, p0=p0, bounds=(0, np.inf))
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        if silent_exceptions:
            if print_info:
                print("   ", repr(e))
            return np.full(len(p0), np.nan), np.full(len(p0), np.nan), np.nan
        else:
            raise e

    model = expfit(ts, *popt)

    if print_info and np.any(np.isfinite(signal-model)):
        names = ["T1", "amp"]
        for p, e, n in zip(popt, perr, names):
            print("   ", n, ufloat(p, e))
        print("    sqrt(resvar)", np.nanvar(signal - model)**0.5)

    if np.any(~np.isfinite(popt)) or np.any(~np.isfinite(perr)):
        if print_info:
                print("    Rejecting because of non-finite result")
        return np.full(popt.size, np.nan), np.full(popt.size, np.nan), np.nan
    if popt[0] > max(ts)*3:
        if print_info:
                print("    Rejecting because T1 longer than 3 times measurement duration")
        return np.full(popt.size, np.nan), np.full(popt.size, np.nan), np.nan

    return popt, perr, model, signal, ts


def fit_Ramsey_chevron_IQdist(res, print_exception=True):
    """Fit (1-exp*cos) for each line in chevron based on distance from ground state.
    Works on RamseyChevron and RamseyChevronRepeat.

    Signal is |Z-Zg|

    Parameters
    ----------
    res : dict
        qmtools result of ramsey_chevron or ramsey_chevron_repeat
    print_exception : bool

    Returns
    -------
    popt : 1D array
        Best values. Columns are: T2 (ns), period (ns), amp (rad), phase (rad).
        Rows may be NaN where fit failed.
    perr : 2D array
        Standard deviations, same columns
    model : 2D array
        Model
    chevron : 2D array
        Data
    ts : 1D array
        Ramsey delay from drive pulse center to center (as in qmtools)
    fdrive : 1D array
        Drive frequencies
    """
    # average repetitions
    if len(res['Z'].shape) == 3:
        S = np.mean(res['Z'], 0)
    # get signal
    chevron = np.abs(S - np.mean(res['Zg']))
    mask = ~np.any(np.isnan(chevron), axis=1)

    # fit Ramsey for each IF
    def ramsey_model(t, t2, period, amp, phase):
        return amp * (1 + np.exp(-t/t2)*np.cos(2*np.pi * t/period + phase))
    ramseymodelbounds = ([2, 0, 0, -np.inf], np.inf)

    ts = np.arange(S.shape[-1])
    popt, perr, model = np.full((chevron.shape[0], 4), np.nan), np.full((chevron.shape[0], 4), np.nan), np.full((chevron.shape[0], ts.size), np.nan)
    for i in np.where(mask)[0]:
        y = chevron[i]
        # initial guess of period based on expected qubit IF, lower bound to 1kHz to avoid singularity
        detuning = max(1e3, np.abs(res['qubitIFs'][i]-res['config']['qubitIF']))
        per0 = min(max(ts), 1e9/detuning)
        # initial parameters
        p0 = [max(ts)/2, per0, np.mean(y), 0]
        try:
            popt[i], pcov = curve_fit(ramsey_model, ts, y, p0, bounds=ramseymodelbounds)
            perr[i] = np.sqrt(np.diag(pcov))
            model[i] = ramsey_model(ts, *popt[i])
        except Exception as e:
            if print_exception:
                print(f"Error in fitting idx {i}:", repr(e))

    return popt, perr, model, chevron, ts, res['qubitIFs']+res['config']['qubitLO']



def fit_Ramsey_chevron_argS(res, min_snr=0, print_exception=True):
    """Fit amp*exp*cos+offs for each line in chevron based on arg S.
    Works on RamseyChevron and RamseyChevronRepeat without ground state information Zg.

    Signal used is argS. Works well if the resonator shift is small or comparable to its width.
    Uses unwrapping, so SNR should be good enough for no random phase wrapping.

    We fit a decaying cosine with a phase (due to detuning) and offset (difference to fitting Z-Zg).

    Parameters
    ----------
    res : dict
        Result dict of QMRamseyChevronRepeat or QMRamseyChevronRepeat_Gaussian
    min_snr : float
        If non-zero, is a minimum bound on SNR per line estimated by smoothing the data.
        Lines (drive frequencies) not meeting this minimum result in NaN parameters.
    print_exception : bool

    Returns
    -------
    popt : 2D array
        Optimal values. Columns are: T2 (ns), period (ns), amp (rad), offset (rad), phase (rad)
    perr : 2D array
        Standard deviations, same columns
    model : 2D array
        Model
    chevron : 2D array
        Data
    ts : 1D array
        Ramsey delay from drive pulse center to center (as in qmtools)
    fdrive : 1D array
        Drive frequencies
    """
    # average over all repetitions
    if len(res['Z'].shape) == 3:
        S = np.mean(res['Z'], 0)
    # Signal is unwrapped phase signal
    chevron = np.unwrap(np.unwrap(np.angle(S), axis=0))
    mask = ~np.any(np.isnan(chevron), axis=1)

    # mask based on rough SNR estimate
    if min_snr > 0:
        chevronsmooth = chevron.copy()
        chevronsmooth[mask] = savgol_filter(chevron[mask], window_length=10, polyorder=2)
        snr = (np.max(chevronsmooth, axis=1) - np.min(chevronsmooth, axis=1)) / np.std(chevron-chevronsmooth, axis=1)
        mask &= (snr >= min_snr)

    # fit Ramsey for each IF
    def ramsey_model(t, t2, period, amp, offset, phase):
        return -amp * np.exp(-t/t2) * np.cos(2*np.pi * t/period + phase) + offset
    ramseymodelbounds = ([2, 0, 0, -np.inf, -np.inf], np.inf)

    ts = np.arange(S.shape[-1])
    popt, perr, model = np.full((chevron.shape[0], 5), np.nan), np.full((chevron.shape[0], 5), np.nan), np.full((chevron.shape[0], ts.size), np.nan)
    for i in np.where(mask)[0]:
        y = chevron[i]
        # initial guess of period based on expected qubit IF, lower bound to 1kHz to avoid singularity
        detuning = max(1e3, np.abs(res['qubitIFs'][i]-res['config']['qubitIF']))
        per0 = min(max(ts), 1e9/detuning)
        p0 = [max(ts)/2, per0, (np.max(y)-np.min(y))/2, np.mean(y), 0]
        try:
            popt[i], pcov = curve_fit(ramsey_model, ts, y, p0, bounds=ramseymodelbounds)
            perr[i] = np.sqrt(np.diag(pcov))
            model[i] = ramsey_model(ts, *popt[i])
        except Exception as e:
            if print_exception:
                print(f"Error in fitting drive idx {i}:", repr(e))

    return popt, perr, model, chevron, ts, res['qubitIFs']+res['config']['qubitLO']


def fit_T1(res, print_info=True):
    """Get T1 from 'relaxation:...' measurement.

    Returns
    -------
    dict
    """
    popt, perr, model, signal, ts = fit_relaxation(res, print_info, silent_exceptions=True)
    return {
        'type': 'T1', 'T': popt[0], 'Terr': perr[0],
        'drive_len': res['drive_len_ns'], 'pi_amp': res['config']['pi_amp'],
        'fdrive': res['config']['qubitIF']+res['config']['qubitLO'],
        'fdriveLO': res['config']['qubitLO']}


def fit_T2(res, min_points=5, print_info=True):
    """Get T2 from 'ramsey_chevron:...' or 'ramsey_chevron_repeat:...' fit decaying cosine.

    If there is Zg information available, uses |Z-Zg| as signal and runs the following constraints:
    Averages over all repetitions in 'ramsey_chevron_repeat' results.

    Fits each drive frequency with a decaying (1+cosine), then masks (removes) data where

    - cosine period longer than 2*T2 (only works when detuned far enough) bc otherwise these parameters a strongly degenerate
    - abs(param) is twice as big as the parameter's error bar for T2 and period,
    - the amplitude is less than a tenth of the maximum amplitude in the data set,
    - the SNR in less than 1

    Notice that the SNR is given by the following components:
    The signal is the oscillation amplitude minus the ground state noise amplitude, i.e. std(|Zg|).
    The noise is the standard deviation of the residuals.

    If there is no Zg information, uses argZ as signal with other constraints,
    because the fit function has the additional parameter offset.

    If there are at least 5 (or min_points) valid lines (drive frequencies), their average parameters is the result, weighted by fit uncertainty.
    The uncertainty is the standard deviation of parameters between lines, so it reflects the variation with drive frequency.

    Returns
    -------
    dict with keys
        type, T, Terr, pi_amp, fdrive, fdriveLO
    """
    result = {
        'type': 'T2*', 'T': np.nan, 'Terr': np.nan,
        'drive_len': res['drive_len_ns'], 'pi_amp': res['config']['pi_amp'],
        'fdrive': res['config']['qubitIF']+res['config']['qubitLO'],
        'fdriveLO': res['config']['qubitLO']}

    if 'Zg' in res:
        popt, perr, model, chevron, ts, fs = fit_Ramsey_chevron_IQdist(res, print_exception=print_info)
        names = ["T2", "detuning", "amp", "phase"]

        # Remove data points with non-finite values or error bars
        mask = ~np.all(np.isfinite(popt), axis=1) | ~np.all(np.isfinite(perr), axis=1)
        # Remove uncorrelated T2 and period
        mask |= (popt[:,1]/2 > popt[:,0])
        # Remove insignificant results compared to errorbars in T2 and amplitude
        mask |= (perr[:,0]*2 > np.abs(popt[:,0])) | (perr[:,1] > np.abs(popt[:,1]))
        # Remove small amplitudes compared to max amplitude (by hard-coded factor 10)
        mask |= (popt[:,2] < np.nanmax(popt[:,2])/10) 

        # SNR estimate:
        noise = np.nanstd(model-chevron, axis=1)
        # Ground state noise
        Zgnoise = np.nanstd(np.abs(res['Zg']-np.mean(res['Zg']))) # each Zg point has same averaging as one point in rest of data
        # Remove SNR less than 1 per point (usually better known for whole line, so this works)
        mask |= ((popt[:,2]-Zgnoise)/noise < 1) # minimum snr per point
    else:
        popt, perr, model, chevron, ts, fs = fit_Ramsey_chevron_argS(res, print_exception=print_info)
        names = ["T2", "detuning", "amp", "offset", "phase"]

        # Remove data points with non-finite values or error bars
        mask = ~np.all(np.isfinite(popt), axis=1) | ~np.all(np.isfinite(perr), axis=1)
        # Remove uncorrelated T2 and period
        mask |= (popt[:,1]/2 > popt[:,0]) #| (popt[:,0] > 1*max(res['delay_ns']))
        # Remove insignificant results compared to errorbars in T2 and amplitude
        mask |= (perr[:,0]*2 > np.abs(popt[:,0])) | (perr[:,1] > np.abs(popt[:,1]))
        # Remove extreme T2
        mask |= (popt[:,0] > 3*np.max(ts))
        # Remove extreme amplitudes
        mask |= (popt[:,2] > 2*(np.max(chevron, axis=1)-np.min(chevron, axis=1)))
        # Remove small amplitudes compared to max amplitude (by hard-coded factor 10)
        if np.count_nonzero(~mask) > 0:
            mask |= (popt[:,2] < np.nanmax(popt[~mask,2])/10)

        # Remove SNR less than 1 per point (usually better known for whole line, so this works)
        noise = np.nanstd(model-chevron, axis=1)
        mask |= popt[:,2]/noise < 1

    if np.count_nonzero(~mask) < min_points:
        if print_info:
            print(f"Not enough good points {np.count_nonzero(~mask)} < {min_points}")
        return result

    # Mean best values based on weighted mean
    mean = np.nansum(popt[~mask]/perr[~mask]**2, axis=0) / np.nansum(1/perr[~mask]**2, axis=0)
    #std = np.sqrt(1/np.nansum(1/perr[~mask]**2, axis=0))
    err = np.nanstd(popt[~mask], axis=0)

    if print_info:
        for i, n in enumerate(names):
            print("   ", n, ufloat(mean[i], err[i]))
        print("    sqrt(resvar)", np.nanvar(chevron - model)**0.5)

    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(err)):
        if print_info:
            print("No finite error bars on mean results, empty result")
        return result

    result['T'] = mean[0]
    result['Terr'] = err[0]
    return result
