"""
Plot all results of a pipeline for one gate point.

If interactive you can move through the dataset::

    %matplotlib widget
    from ipywidgets import interact, IntSlider

    plotter = VgatePipelinePlotter(Vgpl, fqestimate=fq_estimate)
    plotter.make_plot()
    interact(plotter.update_plot, idx=IntSlider(min=0, max=plotter.get_N()-1));

### TODO

- Consistent axis titles and colorbars
- Don't use ax as parameter for update, because it anyways keeps track
  of other artists internally.

"""
import numpy as np
import matplotlib.pyplot as plt
import warnings

def opx_amp2pow(amp, ext_gain_db=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return 10*np.log10(amp**2 * 10) + ext_gain_db
def opx_pow2amp(pow, ext_gain_db=0):
    return np.sqrt(10**((pow-ext_gain_db)/10) / 10)


class VgatePipelinePlotter:
    def __init__(self, Vgpl, fqestimate=None, expnames=None):
        self.Vgpl = Vgpl
        self.data = Vgpl.data
        self.Vgate = Vgpl.Vgate
        self.results = Vgpl.results
        self.barefreq = Vgpl.data['bareIF'][()] if 'bareIF' in Vgpl.data else np.nan
        self.fqestimate = fqestimate
        self.expnames = expnames

    def get_N(self):
        return self.data['Vgate'].size

    def make_plot(self):
        Vgate = self.Vgate
        results = self.results
        fr = self.data['resonatorfit'][:,0,0]

        # list non-empty experiment results
        if self.expnames is None:
            expnames = [k for k,v in results.items() if not all(r is None for r in v) and not k == 'mixer_calibration']
        else:
            expnames = self.expnames
        # distribute in cols and rows
        nexp = len(expnames) + 1
        ncols = self.ncols = int(np.ceil(nexp**0.5))
        nrows = self.nrows = int(np.ceil(nexp/ncols))
        fig, axs = self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, layout='constrained', figsize=(10, 8))

        axs[0,0].set_title('fr vs Vgate')
        axs[0,0].plot(Vgate, fr, '.-')
        self.Vgateline = axs[0,0].axvline(Vgate[0], color='gray', zorder=0)

        self.plotters = []
        self.meta = {n: {} for n in expnames} # dict for persistent data between updates
        for iexp, expname in enumerate(expnames):
            axidx = (iexp+1)%nrows, int((iexp+1) / nrows)
            axs[axidx].set_title(expname, fontsize=8)
            firstidx = [i for i, r in enumerate(results[expname]) if r is not None][0]
            try:
                funcname = expname.split(':')[0]
                getattr(self, 'plt_'+funcname)(expname, axs[axidx], firstidx)
                self.plotters.append((expname, axs[axidx], getattr(self, 'update_'+funcname)))
            except Exception as e:
                print(expname, repr(e))
                pass

        for ax in axs.flat:
            if ax.title._text == '':
                ax.axis('off')

        self.update_plot(0)
        return fig

    def update_plot(self, idx):
        self.Vgateline.set_xdata([self.Vgate[idx]])
        for name, ax, func in self.plotters:
            try:
                func(name, ax, idx)
            except Exception as e:
                print(name, e.__class__, repr(e))
        self.fig.suptitle(f'{idx+1} Vgate={np.round(self.Vgate[idx], 7)}V')
        self.fig.canvas.draw_idle()


    def plt_resonator(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        x = self.meta[name]['x'] = res['resonatorIFs'] / 1e6
        self.meta[name]['line'], = ax.plot(x, np.abs(res['Z']))

    def update_resonator(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['line'].set_ydata(np.full(self.meta[name]['x'].size, np.nan))
        else:
            self.meta[name]['line'].set_ydata(np.abs(res['Z']))

        ax.set_xlabel('resonator IF / MHz')


    def plt_resonator_noise(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        x = self.meta[name]['x'] = res['fftfreq']
        self.meta[name]['line'], = ax.plot(x, np.abs(res['fft']))
        ax.set_xlabel('Frequency / Hz')
        ax.set_ylabel('|FFT S|')
        ax.set_xlim(-300, 300)

    def update_resonator_noise(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['line'].set_ydata(np.full(self.meta[name]['x'].size, np.nan))
        else:
            self.meta[name]['line'].set_data(res['fftfreq'], np.abs(res['fft']))


    def plt_resonator_P1(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        ifs = res['resonatorIFs']
        powers = res['readout_amps']
        xx, yy = np.meshgrid(ifs/1e9, powers, indexing='ij')
        self.meta[name]['xx'] = xx
        self.meta[name]['img'] = ax.pcolormesh(xx, yy, np.full(xx.shape, np.nan))
        ax.set_xlabel('IF / MHz')
        ax.set_ylabel('amplitude')
        ax.set_yscale('log')

    def update_resonator_P1(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['img'].set_array(np.full(self.meta[name]['xx'].shape, np.nan))
        else:
            self.meta[name]['img'].set_array(np.unwrap(np.abs(res['Z'])) / res['readout_amps'][None,:])
            self.meta[name]['img'].autoscale()

    def plt_resonator_excited(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        x = self.meta[name]['x'] = res['resonatorIFs'] / 1e6
        self.meta[name]['line_exc'], = ax.plot(x, np.abs(res['Z'][:,0]), label="Excited state")
        self.meta[name]['line_gnd'], = ax.plot(x, np.abs(res['Z'][:,1]), label="Ground state")

    def update_resonator_excited(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['line_exc'].set_ydata(np.full(self.meta[name]['x'].size, np.nan))
            self.meta[name]['line_gnd'].set_ydata(np.full(self.meta[name]['x'].size, np.nan))
        else:
            self.meta[name]['line_exc'].set_ydata(np.abs(res['Z'][:,0]))
            self.meta[name]['line_gnd'].set_ydata(np.abs(res['Z'][:,1]))
            ax.relim(), ax.autoscale(), ax.autoscale_view()

        ax.legend()
        ax.set_xlabel('resonator IF / MHz')


    def plt_readoutSNR(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        ifs = res['resonatorIFs']
        powers = res['readout_amps']
        xx, yy = np.meshgrid(ifs/1e9, powers, indexing='ij')
        self.meta[name]['xx'] = xx
        self.meta[name]['img']= ax.pcolormesh(xx, yy, np.full(xx.shape, np.nan))
        ax.set_xlabel('IF / MHz')
        ax.set_ylabel('amplitude')
        ax.set_yscale('log')

    def update_readoutSNR(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['img'].set_array(np.full(self.meta[name]['xx'].shape, np.nan))
        else:
            dist = np.abs(res['Z'][...,1] - res['Z'][...,0])
            Zsingle = res['I_single_shot'] + 1j * res['Q_single_shot']
            snr = dist / np.std(Zsingle, axis=0)[:,:,0]
            self.meta[name]['img'].set_array(snr)
            self.meta[name]['img'].autoscale()


    def plt_readoutSNR_P1(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        power = self.meta[name]['readout_power'] = res['readout_power']
        self.meta[name]['line'], = ax.plot(power, res.get('SNR', np.full(power.size, np.nan)))
        ax.set_xlabel('readout power / dBm')
        ax.set_ylabel('amp. SNR')

    def update_readoutSNR_P1(self, name, ax, idx):
        res = self.results[name][idx]
        power = self.meta[name]['readout_power']
        if res is None:
            self.meta[name]['line'].set_ydata(np.full(power.size, np.nan))
        else:
            self.meta[name]['line'].set_ydata(res.get('SNR', np.full(power.size, np.nan)))
            ax.relim(), ax.autoscale(), ax.autoscale_view()

    def plt_qubit(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        self.meta[name]['line'], = ax.plot(res['qubitIFs']/1e6, np.angle(res['Z']))
        ax.set_xlabel('qubit drive IF / MHz')
        ax.set_ylabel('arg S')

    def update_qubit(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['line'].set_data([np.nan], [np.nan])
        else:
            self.meta[name]['line'].set_data(res['qubitIFs']/1e6, np.unwrap(np.angle(res['Z'])))
            ax.relim(), ax.autoscale(), ax.autoscale_view()


    def plt_qubit_P2(self, name, ax, firstidx):
        self.meta[name]['qubitIFrange'] = (
            min(min(r['qubitIFs'])+r['config']['qubitLO'] for r in self.results['qubit_P2'] if r is not None),
            max(max(r['qubitIFs'])+r['config']['qubitLO'] for r in self.results['qubit_P2'] if r is not None))

    def update_qubit_P2(self, name, ax, i):
        ax.clear()
        res = self.results[name][i]
        if res is None:
            #ax.set_title(name)
            return
        qubitLO = res['config']['qubitLO']
        f2 = (res['qubitIFs'] + qubitLO)
        arg = np.unwrap(np.unwrap(np.angle(res['Z']), axis=0))
        p2xx, p2yy = np.meshgrid(res['drive_power'], f2/1e9, indexing='ij')
        ax.pcolormesh(p2xx, p2yy, arg.T)
        qifmin, qifmax = self.meta[name]['qubitIFrange']
        ax.set_ylim(qifmin/1e9, qifmax/1e9)
        # axs[1].set_xscale('log')
        ax.set_xlabel('drive power / dBm')
        ax.set_ylabel('drive freq / GHz')
        ax.axhline(qubitLO/1e9, linestyle='--', color='k', linewidth=0.8, zorder=99)

        # hints
        bareIF = self.barefreq - res['config']['resonatorLO']
        if self.fqestimate is not None:
            frIF = self.data['resonatorfit'][i,0,0]
            if not np.isnan(frIF):
                print(frIF, bareIF, frIF-bareIF)
                ax.axhline(self.fqestimate(frIF - bareIF) / 1e9, color='fuchsia', linewidth=0.8, zorder=99)
        # crosstalk
        ax.axhline((self.barefreq/2)/1e9, color='gray', linewidth=0.8, zorder=99)
        ax.axhline((2*qubitLO - self.barefreq/2)/1e9, color='gray', linewidth=0.8, zorder=99)

        resrabi = self.results['time_rabi'][i]
        if resrabi is not None:
            ax.axhline((resrabi['config']['qubitIF']+resrabi['config']['qubitLO'])/1e9, linestyle='--', color='r', linewidth=0.8, zorder=100)

        #ax.set_title(name+f"\n )


    def plt_time_rabi(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        duration = self.meta[name]['duration'] = res['duration_ns']
        self.meta[name]['line'], = ax.plot(duration, np.angle(res['Z']))
        ax.set_xlabel('drive duration / ns')

    def update_time_rabi(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['line'].set_ydata(np.full(self.meta[name]['duration'].size, np.nan))
        else:
            self.meta[name]['line'].set_ydata(np.unwrap(np.angle(res['Z'])))
            ax.relim(), ax.autoscale(), ax.autoscale_view()


    def plt_time_rabi_chevrons(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        rabi_ifs = res['qubitIFs']
        rabi_durations = res['duration_ns']
        rabixx, rabiyy = np.meshgrid(rabi_durations, rabi_ifs/1e6, indexing='ij')
        self.meta[name]['xx'] = rabixx
        self.meta[name]['img'] = img = ax.pcolormesh(rabixx, rabiyy, np.full(rabixx.shape, np.nan))
        self.meta[name]['line'] = ax.axhline(res['config']['qubitIF']/1e6, linestyle='--', color='r', linewidth=0.8, zorder=99)
        ax.set_xlabel('duration / ns')
        ax.set_ylabel('IF / MHz')
        ax.set_title(name + f"\n qubitLO={res['config']['qubitLO']/1e9:.2}GHz", fontsize=8)
        ax.get_figure().colorbar(img, ax=ax, orientation='horizontal', label='|Z - Z0|')

    def update_time_rabi_chevrons(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['img'].set_array(np.full(self.meta[name]['xx'].shape, np.nan))
            ax.set_title(name, fontsize=8)
        else:
            #signal = np.unwrap(np.unwrap(np.angle(res['Z'])), axis=0)
            signal = np.abs(res['Z'] - np.mean(res['Z'][:,0]))
            self.meta[name]['img'].set_array(signal.T)
            self.meta[name]['img'].autoscale()
            self.meta[name]['line'].set_ydata([res['config']['qubitIF']/1e6])
            ax.set_title(
                name + f"\n qubitLO={res['config']['qubitLO']/1e9:.2}GHz cooldown {res['config']['cooldown_clk']*4/1000:.0f}us"
                f"\n read {opx_amp2pow(res['config']['short_readout_amp']):+.1f}{res['config']['resonator_output_gain']:+.1f}dBm"
                f" drive {opx_amp2pow(res['config']['saturation_amp']):+.1f}{res['config']['qubit_output_gain']:+.1f}dBm", fontsize=8)


    def plt_power_rabi(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        amps = self.meta[name]['amps'] = res['drive_amps']
        self.meta[name]['line'], = ax.plot(amps, np.angle(res['Z']))
        ax.set_xlabel('drive amp / V')

    def update_power_rabi(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['line'].set_ydata(np.full(self.meta[name]['amps'].size, np.nan))
        else:
            self.meta[name]['line'].set_ydata(np.unwrap(np.angle(res['Z'])))
            ax.relim(), ax.autoscale(), ax.autoscale_view()


    def plt_relaxation(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        x = self.meta[name]['x'] = res['delay_ns']
        self.meta[name]['line'], = ax.plot(x, np.abs(res['Z']-res['Z'][0]))
        ax.set_xlabel('delay / ns')

    def update_relaxation(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['line'].set_ydata(np.full(self.meta[name]['x'].size, np.nan))
            ax.set_title(name)
        else:
            signal = np.abs(res['Z'] - res['Z'][0])
            self.meta[name]['line'].set_ydata(signal)
            ax.relim(), ax.autoscale(), ax.autoscale_view()
            ax.set_title(f"{name}\nqubitIF {res['config']['qubitIF']/1e6:.1f}MHz\n"
                         f"{res['drive_len_ns']:.0f}ns pulse {res['config']['pi_amp']:.5f}V{res['config']['qubit_output_gain']:+.1f}dB", fontsize=8)


    def plt_ramsey(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        x = self.meta[name]['x'] = res['delay_ns']
        self.meta[name]['line'], = ax.plot(x, np.unwrap(np.angle(res['Z'])))
        ax.set_xlabel('delay / ns')

    def update_ramsey(self, name, ax, idx):
        res = self.results[name][idx]
        if res is None:
            self.meta[name]['line'].set_ydata(np.full(self.meta[name]['x'].size, np.nan))
            ax.set_title(name)
        else:
            self.meta[name]['line'].set_ydata(np.unwrap(np.angle(res['Z'])))
            ax.relim(), ax.autoscale(), ax.autoscale_view()
            ax.set_title(f"{name}\nqubitIF {res['config']['qubitIF']/1e6:.1f}MHz\n"
                         f"{res['drive_len_ns']:.0f}ns pulse {res['config']['pi_amp']/2:.5f}V{res['config']['qubit_output_gain']:+.1f}dB", fontsize=8)


    def plt_ramsey_repeat(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        x = self.meta[name]['x'] = res['delay_ns']
        ax.set_xlabel('delay / ns')

    def update_ramsey_repeat(self, name, ax, idx):
        res = self.results[name][idx]
        ax.clear()
        ax.set_xlabel('delay / ns')
        if res is None:
            ax.set_title(name)
        else:
            ax.plot(res['delay_ns'], np.unwrap(np.angle(res['Z'])).T)
            ax.relim(), ax.autoscale(), ax.autoscale_view()
            ax.set_title(f"{name}\nqubitIF {res['config']['qubitIF']/1e6:.1f}MHz\n"
                         f"{res['drive_len_ns']:.0f}ns pulse {res['config']['pi_amp']/2:.5f}V{res['config']['qubit_output_gain']:+.1f}dB", fontsize=8)


    def plt_ramsey_chevron_repeat(self, name, ax, firstidx):
        res = self.results[name][firstidx]
        self.meta[name]['line'] = ax.axhline(res['config']['qubitIF']/1e6, linestyle='--', color='r', linewidth=0.8, zorder=99)
        ax.set_xlabel('delay / ns')
        ax.set_ylabel('drive IF / MHz')
        ax.set_title(name + f"\n qubitLO={res['config']['qubitLO']/1e9:.2}GHz", fontsize=8)
        #ax.get_figure().colorbar(img, ax=ax, orientation='horizontal').set_label('arg S')

    def update_ramsey_chevron_repeat(self, name, ax, idx):
        res = self.results[name][idx]
        try:
            self.meta[name]['img'].remove()
        except:
            pass
        if res is None:
            ax.set_title(name, fontsize=8)
        else:
            ifs, ts = res['qubitIFs'], res['delay_ns']
            if 'Zg' in res:
                signal = np.abs(np.mean(res['Z'], axis=0) - np.mean(res['Zg']))
                signaltext = '|Z-Zg|'
            else:
                signal = np.unwrap(np.unwrap(np.angle(np.mean(res['Z'], axis=0)), axis=0))
                signaltext = 'argS'
            xx, yy = np.meshgrid(ts, ifs/1e6, indexing='ij')
            self.meta[name]['xx'] = xx
            self.meta[name]['img'] = img = ax.pcolormesh(xx, yy, signal.T)
            self.meta[name]['line'].set_ydata([res['config']['qubitIF']/1e6])
            ax.set_ylim(np.min(ifs/1e6), np.max(ifs/1e6))
            ax.set_title(
                name + f"\n qubitLO={res['config']['qubitLO']/1e9:.2}GHz cooldown {res['config']['cooldown_clk']*4/1000:.0f}us"
                f"\nread {opx_amp2pow(res['config']['short_readout_amp']):+.1f}{res['config']['resonator_output_gain']:+.1f}dBm"
                f"\ndrive {res['config']['pi_amp']/2:+.5f}V {res['config']['qubit_output_gain']:+.1f}dBm"
                f"\nColor: {signaltext}", fontsize=8)


    def plt_ramsey_anharmonicity(self, name, ax, firstidx):
        return self.plt_ramsey_chevron_repeat(name, ax, firstidx)
    def update_ramsey_anharmonicity(self, name, ax, idx):
        return self.update_ramsey_chevron_repeat(name, ax, idx)

    def plt_hahn_echo(self, name, ax, firstidx):
        return self.plt_ramsey_chevron_repeat(name, ax, firstidx)
    def update_hahn_echo(self, name, ax, idx):
        return self.update_ramsey_chevron_repeat(name, ax, idx)
