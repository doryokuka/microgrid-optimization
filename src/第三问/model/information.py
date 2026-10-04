from bridge import *

def residual_library(centers, actual_net, day, slot, block):
    """Only issued forecasts whose target block has fully elapsed may enter."""
    offset = HOURS[slot]*6
    now = day*144+offset
    ids = [j for j in range(7,day) if j*144+offset+(block+1)*144 <= now]
    rows=[]
    for j in ids:
        start=j*144+offset+block*144
        rows.append(actual_net[start:start+144]-centers[j,slot,block*144:(block+1)*144])
    return np.array(rows), ids
