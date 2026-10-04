"""Portable imports for the unchanged Q3 solvers."""
from common import *
from settlement import dispatch
from continuation import value_point
HOURS=(0,6,12,18)
def timestamp(step):return str(np.datetime64('2025-01-01T00:00','m')+np.timedelta64(int(step)*10,'m'))
