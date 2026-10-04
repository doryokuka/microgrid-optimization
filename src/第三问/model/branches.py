from common import *
from bridge import timestamp as stamp
def date(day):return str(DATES[day])

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def mature_residuals(past_centers, actual_prefix, day, current_slot, history_slot, block):
    """Maturity is checked at the current issue, never the anticipated issue."""
    now = day * 144 + current_slot * 36
    assert past_centers.shape == (day, 4, 288)
    assert actual_prefix.shape == (now,)
    assert history_slot in range(4) and block in (0, 1)
    offset = history_slot * 36
    ids = [h for h in range(7, day) if h * 144 + offset + (block + 1) * 144 <= now]
    assert len(ids) >= 5
    residuals = np.array([
        actual_prefix[h * 144 + offset + block * 144:h * 144 + offset + (block + 1) * 144]
        - past_centers[h, history_slot, block * 144:(block + 1) * 144]
        for h in ids
    ])
    assert residuals.shape == (len(ids), 144) and np.isfinite(residuals).all()
    return residuals, ids

def build_release(day, slot, current_center, past_centers, actual_prefix, price):
    """Build future information branches from a physically restricted API.

    ``price`` has already been rolled to the CURRENT issue, exactly as in Q3.
    Returned baseline paths are an independent Q80 reconstruction for audit.
    """
    assert 31 <= day < 365 and slot in (0, 1, 2)
    now = day * 144 + slot * 36
    assert current_center.shape == (288,)
    assert past_centers.shape == (day, 4, 288)
    assert actual_prefix.shape == (now,)
    assert price.shape == (144,) and np.isfinite(price).all() and np.all(price > 0)
    assert np.isfinite(current_center).all()
    allids = np.arange(7, day)
    revisions = past_centers[allids, slot + 1, :108] - past_centers[allids, slot, 36:144]
    assert np.isfinite(revisions).all()
    scores = revisions @ price[36:144]
    groups = np.array_split(np.argsort(scores, kind='stable'), 3)
    representatives = np.array([revisions[g].mean(axis=0) for g in groups])
    weights = np.array([len(g) / len(allids) for g in groups])
    risks = []
    histories = {}
    for label, history_slot, block in [('current_first24', slot, 0), ('current_second24', slot, 1),
                                       ('next_first24_at_current_issue', slot + 1, 0)]:
        residuals, ids = mature_residuals(past_centers, actual_prefix, day, slot, history_slot, block)
        risks.append(np.quantile(residuals, .8, axis=0, method='linear'))
        histories[label] = dict(history_day_ids=ids, history_size=len(ids), residual_sha256=digest(residuals),
                                historical_issue_hour=history_slot * 6, block=block,
                                last_target_end=stamp(ids[-1] * 144 + history_slot * 36 + (block + 1) * 144),
                                cutoff=stamp(now))
    baseline_n0 = current_center[:144] + risks[0]
    baseline_n1 = current_center[144:] + risks[1]
    n0 = np.repeat(baseline_n0[None, :], 3, axis=0)
    n0[:, 36:] = current_center[36:144] + representatives + risks[2][None, :108]
    n1 = np.repeat(baseline_n1[None, :], 3, axis=0)
    assert np.isfinite(n0).all() and np.isfinite(n1).all()
    assert abs(weights.sum() - 1) < 1e-12
    provenance = dict(day=day, date=date(day), current_release_hour=slot * 6, next_release_hour=(slot + 1) * 6,
        information_cutoff=stamp(now), revision_history_day_ids=allids.tolist(),
        revision_latest_available_issue=stamp(int(allids[-1]) * 144 + (slot + 1) * 36),
        revision_target_start=stamp(now + 36), revision_target_end=stamp(now + 144),
        revision_history_sha256=digest(revisions),
        branches=[dict(name=name, history_day_ids=allids[g].tolist(), count=len(g), weight=float(weights[j]),
                       signed_price_weighted_revision_mean=float(scores[g].mean()),
                       signed_price_weighted_revision_min=float(scores[g].min()),
                       signed_price_weighted_revision_max=float(scores[g].max()),
                       representative_sha256=digest(representatives[j]))
                  for j, (name, g) in enumerate(zip(('low', 'mid', 'high'), groups))],
        risk_histories=histories, current_center_sha256=digest(current_center),
        n0_sha256=digest(n0), n1_sha256=digest(n1), shared_prefix_sha256=digest(baseline_n0[:36]),
        unavailable_future_forecast_passed_to_builder=False, unavailable_actual_passed_to_builder=False,
        actual_prefix_length=len(actual_prefix), first6_all_branches_identical=True,
        second24_all_branches_identical=True)
    return n0, n1, weights, provenance, baseline_n0, baseline_n1, representatives
