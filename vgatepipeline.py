# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

from .helpers import plt2dimg
from .pipeline_fits import fit_T1, fit_T2

class VgatePipeline:
    """
    Represents data from running the QM pipeline vs gate voltage.

    Provides utility functions to gather program outputs into arrays.
    """

    def __init__(self, fpath):
        self.data = np.load(fpath, allow_pickle=True)
        self.results = self.data['results'][()]
        self.Vgate = self.data['Vgate']

    def get_Vgate(self):
        return self.Vgate

    def non_empty_result_keys(self):
        """List all result keys for which at least one gate point contains data.
        The mixer_calibration is excluded, as it doesn't produce data.

        Returns
        -------
        List of str
        """
        return [
            k for k, v in self.results.items()
            if not all(r is None for r in v) and not k == 'mixer_calibration']

    def get_shared_value(self, resultkey, variablekey):
        """
        Return value of a variable in a result if it is the same for all results of that key.
        Otherwise raises exception.

        None if all results with that key are None.
        """
        self.results[resultkey] # check whether present
        try:
            values = [
                res[variablekey] for res in self.results[resultkey]
                if res is not None]
        except KeyError:
            raise Exception(f"{variablekey} not present in all results of {resultkey}")
        if len(values) == 0:
            return None
        if not all(np.allclose(values[i], values[0]) for i in range(1, len(values))):
            raise Exception(f"{variablekey} in {resultkey} does not have the same value for all gate points")
        else:
            return values[0]

    def get_shared_config_value(self, resultkey, configpath: list):
        self.results[resultkey]
        values = [reduce(lambda d, k: d[k], configpath, res['config'])
                  for res in self.results[resultkey]
                  if res is not None]
        if len(values) == 0:
            return None
        if not all(v == values[0] for v in values[1:]):
            raise Exception(f"value at {repr(configpath)} not the same for all gate points in config result {resultkey}")
        return values[0]

    def get_value_as_array(self, resultkey, variablekey):
        """
        Return all values of a variable in a result stacked along first axis.
        First axis has length of Vgate.
        """
        self.results[resultkey] # check whether present
        values = [
            res[variablekey] if res is not None else np.nan
            for res in self.results[resultkey]]
        shapes = [v.shape for v in values if v is not np.nan and hasattr(v, 'shape')]
        if len(shapes) == 0:
            return np.stack(values)
        if not all(s == shapes[0] for s in shapes[1:]):
            raise Exception(f"data in variable {variablekey} in result {resultkey} does not have the same shape for all gate points")
        values = [np.broadcast_to(v, shapes[0]) for v in values]
        return np.stack(values)

    def get_config_value_as_array(self, resultkey, configpath: list):
        self.results[resultkey]
        values = [
            reduce(lambda d, k: d[k], configpath, res['config']) if res is not None else np.nan
            for res in self.results[resultkey]
        ]
        shapes = [v.shape for v in values if v is not np.nan and hasattr(v, 'shape')]
        if len(shapes) == 0:
            return np.stack(values)
        if not all(s == shapes[0] for s in shapes[1:]):
            raise Exception(f"data in config {'.'.join(configpath)} in result {resultkey} does not have the same shape for all gate points")
        values = [np.broadcast_to(v, shapes[0]) for v in values]
        return np.stack(values)


    def merge_qubit_P2(self, resultkey='qubit_P2', flatargS=False, powerdecimals=2):
        """
        Merge two-tone measurements of type `qubit_P2` with different
        LO/IF frequencies per gate point into one array with absolute frequency.

        Parameters
        ----------
        resultkey : str
            Key in results. Has to be a `qubit_P2` measurement.
        flatargS : bool
            If True converts complex value to argument and removes vertical average.
            The result has the same shape but is of type float instead of complex.
            Default False.
        powerdecimals : int
            Rounds power to this number of decimals before use and comparing.
            Since the power (in dB) is converted from a float log scale with boundaries
            set by hand, they often don't exactly align in all digits.
            It is required that after rounding no power values coincide.
            Default is 2.

        Returns
        -------
        f2 : numpy.ndarray
            Drive frequency axis
        power : numpy.ndarray
            Power values in dBm, output at octave
        Zmerged : numpy.ndarray
            Complex valued scattering parameter with dimensions (Vgate, f2, power).
            If flatargS is True, returns unwrapped arg(Zmerged) after removing column-wise averages.
        """
        assert resultkey == 'qubit_P2' or resultkey.startswith('qubit_P2:')
        fqs = np.sort(np.unique([res['qubitIFs']+res['config']['qubitLO'] for res in self.results[resultkey] if res is not None]))
        powers = np.sort(np.unique([np.round(res['drive_power'], powerdecimals) for res in self.results[resultkey] if res is not None]))
        Zmerged = np.full((self.Vgate.size, fqs.size, powers.size), np.nan+0j)
        for i, res in enumerate(self.results[resultkey]):
            if res is None: continue
            fq = res['qubitIFs'] + res['config']['qubitLO']
            assert np.all(np.sort(fq) == fq)
            fmask = np.any(np.isclose(fqs[:,None], fq[None,:]), axis=1)

            ps = np.round(res['drive_power'], powerdecimals)
            assert np.all(np.sort(ps) == ps)
            assert np.all(np.unique(ps) == ps), "Power in dB closer than given rounding"
            pmask = np.any(np.isclose(powers[:,None], ps[None,:]), axis=1)
            #print(Zmerged.shape, fmask.shape, pmask.shape)
            #print(Zmerged[i][fmask,:][:,pmask].shape, res['Z'].shape)
            # Zmerged[i][fmask,:][:,pmask] = res['Z'] # doesn't work, not modifying Zmerged
            pexpand = np.full((fq.size, powers.size), np.nan+0j)
            pexpand[:,pmask] = res['Z']
            fexpand = np.full((fqs.size, powers.size), np.nan+0j)
            fexpand[fmask,:] = pexpand
            Zmerged[i] = fexpand
        if flatargS:
            argZmerged = np.full(Zmerged.shape, np.nan)
            for i in range(Zmerged.shape[-1]):
                argS = np.angle(Zmerged[:,:,i])
                argSmean = []
                for j in range(argS.shape[0]): # for every column
                    # unwrap
                    argS[j][~np.isnan(argS[j])] = np.unwrap(argS[j][~np.isnan(argS[j])], axis=0)
                    # average over last ten values per gate point
                    v = argS[j][~np.isnan(argS[j])][-10:]
                    argSmean.append(np.mean(v) if len(v) else np.nan)
                # flatten, then centered around argS=0
                argS -= np.array(argSmean)[:,None]
                # rewrap in -pi to +pi
                argS = (argS+np.pi)%(2*np.pi)-np.pi
                argZmerged[:,:,i] = argS
            return fqs, powers, argZmerged
        else:
            return fqs, powers, Zmerged


    def plot_Vgate_2tone_multi(self, keys=None, npowers=3, **figkwargs):
        """Make overview figure with data from multiple 2tone results.

        Requires same drive IFs & LO for all gate points for each result key.

        If no keys are given will plot all 2tone spectroscopy results.
        """
        if keys is None:
            ks = self.non_empty_result_keys()
            keys = [k for k in ks if k == 'qubit_P2' or k.startswith('qubit_P2:')]
        fig, axs = plt.subplots(nrows=len(keys)+1, ncols=npowers, sharex=True, layout='constrained', **figkwargs)
        # plot cavity
        S21 = self.get_value_as_array('resonator', 'Z')
        ifs = self.get_shared_value('resonator', 'resonatorIFs')
        for j in range(npowers):
            im = plt2dimg(axs[0,j], self.Vgate, ifs/1e6, np.abs(S21))
            if j > 0: axs[0,j].sharey(axs[0,0])
        fig.colorbar(im, ax=axs[0,-1], label='|S21|')
        axs[0,0].set_ylabel("readout IF / MHz", fontsize=8)
        axs[0,npowers//2].set_title("Cavity, no drive", fontsize=8)
        # plot
        for i, key in enumerate(keys[::-1]):
            qlo = self.get_shared_config_value(key, ['qubitLO'])
            qfs = self.get_shared_value(key, 'qubitIFs') + qlo
            p2s = self.get_shared_value(key, 'drive_power') # (Vgate, f2, P2)
            Z = self.get_value_as_array(key, 'Z')
            idxs = [int(len(p2s)/max(2,npowers-1)*k) for k in range(npowers)][:-1] + [-1]
            for j, p2idx in enumerate(idxs):
                argZ = np.unwrap(np.unwrap(np.angle(Z[:,:,p2idx]), axis=0))
                argZ -= np.nanmedian(argZ, axis=1)[:,None]
                argZ = (argZ+np.pi)%(2*np.pi)-np.pi # constrain to -pi to +pi
                im = plt2dimg(axs[i+1, j], self.Vgate, qfs/1e9, argZ)
                if j > 0:
                    axs[i+1,j].sharey(axs[i+1,0])
                fig.colorbar(im, ax=axs[i+1,j])#, label="arg Z - <arg Z>_f2", fs=8)
                axs[i+1,j].set_title(f"P2 = {p2s[p2idx]:+.1f}dBm", fontsize=8)
            axs[i+1,0].set_ylabel(f"{key}\nf2 / GHz", fontsize=8)
        axs[-1,npowers//2].set_xlabel("Vgate / V")
        return fig

    def fit_all_coherence(self, result_keys=None, print_info=True):
        """
        Fit T1, T2* and T2E for all gate points.

        Returns
        -------
        dict
            One item for each result key with the fit results. Fit results are dicts
            with keys from fit_T1, fit_T2, ... but all fields converted to arrays
            corresponding to Vgate, except for type, which is just a string.
        """
        results = {}
        keys = self.non_empty_result_keys() if result_keys is None else result_keys
        for key in keys:
            if key == 'relaxation' or key.startswith('relaxation:'):
                if print_info:
                    print(key)
                results[key] = [None]*self.Vgate.size
                for i, res in enumerate(self.results[key]):
                    try:
                        if res is not None:
                            results[key][i] = fit_T1(res, print_info=False)
                    except Exception as e:
                        if print_info:
                            print("Vgate idx {i}:", repr(e))
            elif (key == 'ramsey_chevron' or key.startswith('ramsey_chevron:')
                  or key == 'ramsey_chevron_repeat' or key.startswith('ramsey_chevron_repeat:')):
                if print_info:
                    print(key)
                results[key] = [None]*self.Vgate.size
                for i, res in enumerate(self.results[key]):
                    try:
                        if res is not None:
                            results[key][i] = fit_T2(res, print_info=False)
                    except Exception as e:
                        if print_info:
                            print(f"Vgate idx {i}:", repr(e))

        # Convert (dict of list of dicts) to (dict of dict with array)
        results2 = {}
        for key, fitlist in results.items():
            if all(rec is None for rec in fitlist):
                continue
            fitvaluenames = set(k for rec in fitlist if rec is not None for k in rec.keys())
            fitvaluenames.remove('type') # this gets a singular value instead of a list
            fitresult = {'type': [rec['type'] for rec in fitlist if rec is not None][0]}
            for vn in fitvaluenames:
                values = [
                    fitlist[i][vn] if fitlist[i] is not None and vn in fitlist[i] else np.nan
                    for i in range(self.Vgate.size)]
                fitresult[vn] = np.array(values)
            results2[key] = fitresult
        if len(results2):
            results2['Vgate'] = self.Vgate
        return results2
