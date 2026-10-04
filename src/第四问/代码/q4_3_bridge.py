"""Runtime and physical constants shared with final Q2, no attachment1 reads."""
from run_q4_2 import WORK, SOURCE
from q2.common import *
from q2.settlement import dispatch
from q4_value import value_point as _value_point
HOURS=(0,6,12,18)
def timestamp(step):
    return str(np.datetime64('2025-01-01T00:00','m')+np.timedelta64(int(step)*10,'m'))
def value_point(nets,p,E):
    # Identical equal-weight paths represent the same deterministic tail problem.
    if np.array_equal(nets,np.broadcast_to(nets[:1],nets.shape)):nets=nets[:1]
    return _value_point(nets,p,E,'mean')
