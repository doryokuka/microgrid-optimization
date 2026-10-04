"""Single settlement against the immutable midnight reference, never version deltas."""
from q4_3_bridge import np

def components(q0, executed, emergency, price):
    q0, executed, emergency, price = np.broadcast_arrays(q0, executed, emergency, price)
    up = np.maximum(executed-q0, 0.)
    down = np.maximum(q0-executed, 0.)
    plan = price*q0
    refund = price*down
    penalty = .5*price*down
    addition = 1.5*price*up
    emergency_fee = 5*price*emergency
    ordinary = plan-refund+penalty+addition
    assert np.allclose(ordinary, price*executed+.5*price*np.abs(executed-q0), atol=1e-7)
    return dict(original_plan_fee=plan, cancellation_refund=refund,
                cancellation_penalty=penalty, additional_purchase_fee=addition,
                emergency_fee=emergency_fee, settled_ordinary_fee=ordinary,
                total_fee=ordinary+emergency_fee, increase_kwh=up, decrease_kwh=down)

def planning_fee(q, price, reference):
    """NaN denotes uncommitted future-day procurement, priced at the normal rate."""
    mask = np.isfinite(reference)
    return float(price @ q + .5*np.sum(price[mask]*np.abs(q[mask]-reference[mask])))
