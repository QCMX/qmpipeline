# -*- coding: utf-8 -*-


import numpy as np
import itertools
from deprecation import deprecated


@deprecated(details="Use qualang_tools.loops.from_array")
def int_forloop_values(start, end, step):
    """
    qua.for_(f, f_min, f <= f_max, f + df)
    """
    if int(start) != start:
        print("WARNING int_forloop_values: start value will be rounded")
    if int(end) != end:
        print("WARNING int_forloop_values: end value will be rounded")
    if int(step) != step:
        print("WARNING int_forloop_values: step size will be rounded")
    start, end, step = int(start), int(end), int(step)
    if step == 0:
        raise ValueError("int_forloop_values: step size == 0. Infinite loop")
    if (start < end and step < 0) or (start > end and step > 0):
        raise ValueError("int_forloop_values: step has wrong sign. Infinite loop")
    gen = itertools.takewhile(lambda x: x <= end, itertools.count(start=start, step=step))
    return np.array(list(gen))
