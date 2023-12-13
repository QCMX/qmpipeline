# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from datetime import datetime
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plt2dimg(ax, x, y, data, **kwargs):
    """x along first axis of data"""
    assert data.shape == x.shape + y.shape
    if 'interpolation' not in kwargs:
        kwargs = kwargs.copy()
        kwargs['interpolation'] = 'nearest'
    return ax.imshow(
        data.T, origin='lower', aspect='auto',
        extent=(
            x[0]-(x[1]-x[0])/2,
            x[-1]+(x[-1]-x[-2])/2,
            y[0]-(y[1]-y[0])/2,
            y[-1]+(y[-1]-y[-2])/2),
        **kwargs)


def plt2dimg_update(im, data, rescalev=True):
    im.set_data(data.T)
    if rescalev:
        im.set_clim(vmin=np.nanmin(data), vmax=np.nanmax(data))


def pltcolorline(ax, x, y, **kwargs):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = plt.cm.viridis(np.linspace(0, 1, segments.shape[0]))
    lc = LineCollection(segments, colors=colors, **kwargs)
    ax.autoscale()
    return ax.add_collection(lc)


# https://stackoverflow.com/a/55456635
def mpl_pause(interval):
    """Like matplotlib.pause() but doesn't grab focus for plot window."""
    backend = matplotlib.get_backend()
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
        else:
            time.sleep(interval)
    else:
        raise Exception("mpl backend not interactive")


def data_path(
        fname: str,
        time: datetime = None,
        basepath: str = '.',
        bydate: bool = True,
        datesuffix: str = ''):
    if time is None:
        time = datetime.now()
    date = time.strftime('%Y-%m-%d') + datesuffix
    if bydate:
        dir = os.path.join(basepath, date)
    else:
        dir = basepath
    if not os.path.exists(dir):
        print("Creating directory "+dir)
        os.mkdir(dir)
    return os.path.join(dir, fname.format(datetime=time.strftime('%Y-%m-%d_%H-%M-%S')))


def savez(fpath, **kwargs):
    warn('Use numpy.savez and helpers.data_path instead.',
         DeprecationWarning, stacklevel=2)
    dt = kwargs.get('datetime', datetime.now())
    return np.savez(
        fpath.format(datetime=dt.strftime('%Y-%m-%d_%H-%M-%S')),
        **kwargs)


class DurationEstimator:
    def __init__(self, N):
        """Automatically starts time for the first step()"""
        self.N = N
        self.start()

    def start(self, starti=0):
        """(Re)Start estimation."""
        self.tstart = self.t0 = datetime.now()
        self.dt = None
        self.lasti = 0

    def elapsed(self):
        """Get total time since start in seconds."""
        return (datetime.now() - self.tstart).total_seconds()

    def step(self, i, printinfo=True):
        """Signal one step finished and print info to stdout.

        You may skip calling this function for some i. This even increases
        precision, because implicitly it averages the time of the skipped steps.
        
        If called with the same i again, function does nothing. Especially
        after starting.

        Undefined behavior for not monotonously increasing i.
        """
        if i == self.lasti:
            return
        now = datetime.now()
        if self.dt is None:
            self.dt = (now - self.t0) / (i - self.lasti)
        else:
            self.dt = self.dt * 2/3 + (now - self.t0) / (i - self.lasti) * 1/3
        self.t0 = now
        self.lasti = i
        if printinfo:
            print('{} :  {percentage:.1f} %  (remaining {remaining})'.format(
                now.strftime('%Y-%m-%d %H:%M:%S'),
                percentage=100*i/(self.N-1),
                remaining=(self.N-i)*self.dt))

    def end(self):
        """Print total duration."""
        now = datetime.now()
        print('{} :  end. Total duration: {})'.format(
            now.strftime('%Y-%m-%d %H:%M:%S'), now - self.tstart))
