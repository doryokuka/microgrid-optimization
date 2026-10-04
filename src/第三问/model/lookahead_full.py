import bridge
from scipy.sparse import coo_matrix
import highspy,time
np=bridge.np
LOW,HIGH,ETA,B=bridge.LOW,bridge.HIGH,bridge.ETA,bridge.B
dispatch,value_point=bridge.dispatch,bridge.value_point
NEXT_RELEASE=36
TARGET_GAP=.001
ROOT_LIMIT=2.
MAX_ITER=3

def _weighted_median(values, weights):
    """Deterministic lower weighted median, independently for every interval."""
    order = np.argsort(values, axis=0, kind='stable')
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(
        np.broadcast_to(weights[:, None], values.shape), order, axis=0)
    indices = np.argmax(np.cumsum(sorted_weights, axis=0) >= .5, axis=0)
    return sorted_values[indices, np.arange(values.shape[1])]

def _highs_model(cost, lo, hi, integrality, a, rl, ru):
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(cost), len(rl)
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, lo, hi
    lp.row_lower_, lp.row_upper_ = rl, ru
    lp.integrality_ = [highspy.HighsVarType.kInteger if z else
                       highspy.HighsVarType.kContinuous for z in integrality]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_ = a.indptr, a.indices
    lp.a_matrix_.value_ = a.data
    return lp

def _root(nets, price, s0, weights, cuts, reference=None, seed_q=None, previous=None):
    """Exact frozen B-policy MILP, with its continuous relaxation as a bound."""
    started = time.perf_counter()
    K, T = nets.shape
    S = K*T
    qb, physical = T, T+S
    tail, ab = physical+10*S, physical+10*S+K
    nv = ab+S
    qi = lambda k, t: qb+k*T+t
    ix = lambda typ, k, t: physical+typ*S+k*T+t
    ai = lambda k, t: ab+k*T+t
    qub = np.maximum(0., nets.max(axis=0))+B
    cost = np.zeros(nv)
    lo, hi = np.zeros(nv), np.full(nv, np.inf)
    hi[:T] = qub
    variable_contract = reference is None
    if not variable_contract:
        # Only the prefix of this vector is a decision in update mode. The
        # distant returned plan is constructed for diagnostics after solving.
        hi[NEXT_RELEASE:T] = 0.
    integer = np.zeros(nv, dtype=np.int32)
    rr, cc, vv, rl, ru = [], [], [], [], []

    def row(dic, lower=-np.inf, upper=np.inf):
        r = len(rl)
        for col, val in dic.items():
            rr.append(r); cc.append(col); vv.append(val)
        rl.append(lower); ru.append(upper)

    for k in range(K):
        cost[qb+k*T:qb+(k+1)*T] = weights[k]*price
        hi[qb+k*T:qb+(k+1)*T] = qub
        cost[tail+k] = weights[k]
        for t in range(T):
            q, aq = qi(k, t), ai(k, t)
            if t < NEXT_RELEASE:
                row({q: 1., t: -1.}, 0., 0.)
            if variable_contract and t >= NEXT_RELEASE:
                cost[aq] = weights[k]*.5*price[t]
                row({q: 1., t: -1., aq: -1.}, upper=0.)
                row({q: -1., t: 1., aq: -1.}, upper=0.)
            elif not variable_contract and np.isfinite(reference[t]):
                cost[aq] = weights[k]*.5*price[t]
                row({q: 1., aq: -1.}, upper=float(reference[t]))
                row({q: -1., aq: -1.}, upper=-float(reference[t]))
            else:
                hi[aq] = 0.
            c, u, e, w, s, b, cb, hb, ub, lb = [ix(j, k, t) for j in range(10)]
            em = max(0., nets[k, t])
            wm = max(0., qub[t]-nets[k, t])
            hi[c] = hi[u] = B
            hi[e], hi[w], lo[s], hi[s] = em, wm, LOW, HIGH
            for z in (b, cb, hb, ub, lb):
                hi[z] = 1.; integer[z] = 1
            cost[e] = weights[k]*5*price[t]
            row({q: 1., e: 1., u: 1., c: -1., w: -1.}, nets[k, t], nets[k, t])
            energy = {s: 1., c: -ETA, u: 1/ETA}
            if t:
                energy[ix(4, k, t-1)] = -1.
            row(energy, s0 if t == 0 else 0., s0 if t == 0 else 0.)
            # These are exactly the B-policy constraints from frozen stage.py.
            row({c: 1., b: -B}, upper=0.)
            row({u: 1., b: B}, upper=B)
            row({w: 1., b: -wm}, upper=0.)
            row({e: 1., b: em}, upper=em)
            row({c: -1., cb: B}, upper=0.)
            row({s: -1., hb: HIGH-LOW}, upper=-LOW)
            row({w: 1., cb: -wm, hb: -wm}, upper=0.)
            row({u: -1., ub: B}, upper=0.)
            row({s: 1., lb: HIGH-LOW}, upper=HIGH)
            row({e: 1., ub: -em, lb: -em}, upper=0.)
        for slope, intercept in cuts[k]:
            row({tail+k: 1., ix(4, k, T-1): -slope}, lower=intercept)

    a = coo_matrix((vv, (rr, cc)), shape=(len(rl), nv)).tocsr()
    rl, ru = np.asarray(rl), np.asarray(ru)

    def envelope(k, energy):
        return max(0., max(slope*energy+intercept for slope, intercept in cuts[k]))

    def feasible_vector(q0, branch_q):
        q0 = np.clip(np.asarray(q0, float), 0., qub)
        if not variable_contract:
            q0[NEXT_RELEASE:] = 0.
        branch_q = np.clip(np.asarray(branch_q, float), 0., qub)
        branch_q[:, :NEXT_RELEASE] = q0[None, :NEXT_RELEASE]
        x = np.zeros(nv)
        x[:T], x[qb:physical] = q0, branch_q.ravel()
        for k in range(K):
            r = dispatch(branch_q[k], nets[k], s0)
            x[tail+k] = envelope(k, r['s'][-1])
            if variable_contract:
                x[ab+k*T+NEXT_RELEASE:ab+(k+1)*T] = np.abs(branch_q[k, NEXT_RELEASE:]-q0[NEXT_RELEASE:])
            else:
                finite = np.isfinite(reference)
                x[ab+k*T:ab+(k+1)*T][finite] = np.abs(branch_q[k, finite]-reference[finite])
            for t in range(T):
                vals = (r['c'][t], r['d'][t], r['e'][t], r['w'][t], r['s'][t+1],
                        int(branch_q[k, t] >= nets[k, t]), int(r['c'][t] >= B-1e-7),
                        int(r['s'][t+1] >= HIGH-1e-7), int(r['d'][t] >= B-1e-7),
                        int(r['s'][t+1] <= LOW+1e-7))
                for j, val in enumerate(vals):
                    x[ix(j, k, t)] = val
        return x

    # Relaxing all binary variables gives a valid lower bound for this exact
    # MILP. The resulting plans are merely candidates until independently
    # replayed under B, not assumed to be feasible B schedules.
    relaxed = highspy.Highs()
    relaxed.setOptionValue('output_flag', False)
    relaxed.setOptionValue('threads', 1)
    relaxed.passModel(_highs_model(cost, lo, hi, np.zeros(nv, int), a, rl, ru))
    lp_start = time.perf_counter()
    relaxed.run()
    lp_seconds = time.perf_counter()-lp_start
    assert relaxed.getModelStatus() == highspy.HighsModelStatus.kOptimal, str(relaxed.getModelStatus())
    lp_solution = np.asarray(relaxed.getSolution().col_value)
    lp_bound = float(relaxed.getInfo().objective_function_value)
    lp_q0, lp_branch = lp_solution[:T].copy(), lp_solution[qb:physical].reshape(K, T).copy()
    candidates = [(lp_q0, lp_branch)]
    median_q0 = _weighted_median(lp_branch, weights)
    median_q0[:NEXT_RELEASE] = lp_q0[:NEXT_RELEASE]
    candidates.append((median_q0, lp_branch))
    naive_branch = np.clip(nets, 0., qub)
    naive_q0 = _weighted_median(naive_branch, weights)
    candidates.append((naive_q0, naive_branch))
    if seed_q is not None:
        seed_q = np.clip(np.asarray(seed_q, float), 0., qub)
        assert seed_q.shape == (T,)
        candidates.append((seed_q, np.tile(seed_q, (K, 1))))
        candidates.append((seed_q, naive_branch))
    if previous is not None:
        candidates.append(previous)
    xs = [feasible_vector(q0, qj) for q0, qj in candidates]
    x_seed = min(xs, key=lambda x: cost@x)
    seed_value = float(cost@x_seed)
    ax = a@x_seed
    violation = max(float(np.maximum(rl-ax, 0).max()), float(np.maximum(ax-ru, 0).max()),
                    float(np.maximum(lo-x_seed, 0).max()), float(np.maximum(x_seed-hi, 0).max()))
    assert violation < 1e-5, ('B warm start infeasible', violation)
    seed_gap = max(0., (seed_value-lp_bound)/max(abs(seed_value), 1e-12))
    raw_gap = None
    milp_replay_error, milp_objective_error = None, None
    if seed_gap <= TARGET_GAP:
        chosen = x_seed
        lower_bound = lp_bound
        status, solver_seconds = 'Certified_B_replay_against_LP_bound', 0.
    else:
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        h.setOptionValue('threads', 1)
        h.setOptionValue('time_limit', ROOT_LIMIT)
        h.setOptionValue('mip_rel_gap', TARGET_GAP)
        h.setOptionValue('random_seed', 2026)
        h.passModel(_highs_model(cost, lo, hi, integer, a, rl, ru))
        h.setSolution(nv, np.arange(nv, dtype=np.int32), x_seed)
        solve_start = time.perf_counter()
        h.run()
        solver_seconds = time.perf_counter()-solve_start
        info, solution = h.getInfo(), h.getSolution()
        lower_bound = max(lp_bound, float(info.mip_dual_bound))
        raw_gap = float(info.mip_gap)
        status = str(h.getModelStatus())
        chosen = x_seed
        if solution.value_valid:
            raw = np.asarray(solution.col_value)
            candidate = feasible_vector(raw[:T], raw[qb:physical].reshape(K, T))
            milp_replay_error = float(np.max(np.abs(candidate[physical:physical+5*S]-raw[physical:physical+5*S])))
            milp_objective_error = abs(float(cost@candidate)-float(info.objective_function_value))
            assert milp_replay_error < 2e-3 and milp_objective_error < 1e-2, (milp_replay_error, milp_objective_error)
            # Replay, not the MILP's numerical objective, selects the incumbent.
            if cost@candidate < cost@chosen:
                chosen = candidate
    q0, branch_q = chosen[:T].copy(), chosen[qb:physical].reshape(K, T).copy()
    incumbent = float(cost@chosen)
    assert lower_bound <= incumbent+max(1e-4, 1e-8*abs(incumbent)), (lower_bound, incumbent)
    final_ax = a@chosen
    final_violation = max(float(np.maximum(rl-final_ax, 0).max()), float(np.maximum(final_ax-ru, 0).max()),
                          float(np.maximum(lo-chosen, 0).max()), float(np.maximum(chosen-hi, 0).max()))
    assert final_violation < 1e-5, ('Final B replay infeasible', final_violation)
    trajectories = [dispatch(branch_q[k], nets[k], s0) for k in range(K)]
    replay_error = 0.
    for k, r in enumerate(trajectories):
        for j, key in enumerate(('c', 'd', 'e', 'w', 's')):
            truth = r[key][1:] if key == 's' else r[key]
            saved = chosen[physical+j*S+k*T:physical+j*S+(k+1)*T]
            replay_error = max(replay_error, float(np.max(np.abs(truth-saved))))
    stats = dict(continuous=int(nv-integer.sum()), binary=int(integer.sum()), constraints=len(rl),
                 incumbent=incumbent, lower_bound=lower_bound,
                 gap=max(0., (incumbent-lower_bound)/max(abs(incumbent), 1e-12)),
                 raw_mip_gap=raw_gap, relaxation_lower_bound=lp_bound,
                 seconds=solver_seconds, relaxation_seconds=lp_seconds,
                 total_seconds=time.perf_counter()-started, status=status,
                 replay_max_error=replay_error, warmstart_constraint_violation=violation,
                 final_constraint_violation=final_violation, milp_replay_max_error=milp_replay_error,
                 milp_objective_error=milp_objective_error,
                 seed_incumbent=seed_value, seed_gap=seed_gap)
    if not variable_contract:
        q0[NEXT_RELEASE:] = _weighted_median(branch_q[:, NEXT_RELEASE:], weights)
    return q0, branch_q, stats, trajectories

def solve_lookahead(n0, n1, price, s0, weights, reference=None, seed_q=None):
    """Return common executable prefix, audited 48h gap, and branch plans.

    reference=None creates a midnight contract. A supplied vector is an
    immutable original-contract reference (NaN means uncommitted next day).
    In update mode returned entries 36: are branch weighted-median diagnostic
    values, not a new contract or a binding common recourse commitment.

    n0/n1 have shape (3,144). All n0[:,:36] must agree. The historical
    information builder, outside this solver, enforces the causal cutoff.
    The tail uses the frozen value_point with each deterministic branch
    repeated five times, preserving its original K=5 fee normalization.
    """
    n0, n1, price, weights = map(lambda x: np.asarray(x, float), (n0, n1, price, weights))
    assert n0.shape == n1.shape == (3, 144)
    assert price.shape == (144,) and np.all(price > 0.)
    assert weights.shape == (3,) and np.all(weights > 0.) and abs(weights.sum()-1.) < 1e-10
    assert np.isfinite(n0).all() and np.isfinite(n1).all()
    assert np.max(np.abs(n0[:, :NEXT_RELEASE]-n0[0, :NEXT_RELEASE])) < 1e-10
    assert LOW-1e-8 <= s0 <= HIGH+1e-8
    if reference is not None:
        reference = np.asarray(reference, float).copy()
        assert reference.shape == (144,)
        assert np.all(np.isfinite(reference) | np.isnan(reference))
        assert np.all(reference[np.isfinite(reference)] >= 0.)
    start = time.perf_counter()
    cache = {}
    keys = [n1[k].tobytes() for k in range(3)]

    def point(k, energy):
        # Exact energies, not rounded states: cached bounds retain their origin.
        key = (keys[k], float(energy))
        if key not in cache:
            cache[key] = value_point(np.repeat(n1[k:k+1], 5, axis=0), price, float(energy))
        return cache[key]

    grid = np.arange(LOW, HIGH+1., 1200.)
    points = [[point(k, e) for e in grid] for k in range(3)]
    iterations, best = [], None
    global_lower = -np.inf
    for iteration in range(MAX_ITER):
        cuts = [[(z['slope'], z['intercept']) for z in points[k]] for k in range(3)]
        previous = None if best is None else (best['q0'], best['branch_q'])
        q0, branch_q, root, replay = _root(n0, price, s0, weights, cuts, reference=reference, seed_q=seed_q, previous=previous)
        terminal = np.array([r['s'][-1] for r in replay])
        soc6 = np.array([r['s'][NEXT_RELEASE] for r in replay])
        nxt = [point(k, terminal[k]) for k in range(3)]
        ordinary = branch_q@price
        if reference is None:
            adjustment = .5*np.abs(branch_q[:, NEXT_RELEASE:]-q0[None, NEXT_RELEASE:])@price[NEXT_RELEASE:]
        else:
            finite = np.isfinite(reference)
            adjustment = .5*np.abs(branch_q[:, finite]-reference[None, finite])@price[finite]
        emergency = np.array([5*price@r['e'] for r in replay])
        first24 = float(weights@(ordinary+adjustment+emergency))
        tail_values = np.array([z['value'] for z in nxt])
        # Independently replay every full tail; no lower envelope is an upper.
        tail_q = np.array([z['q'] for z in nxt])
        tail_replay = np.array([price@tail_q[k]+5*price@dispatch(tail_q[k], n1[k], terminal[k])['e'] for k in range(3)])
        assert np.max(np.abs(tail_replay-tail_values)) < 1e-3
        upper = float(first24+weights@tail_replay)
        global_lower = max(global_lower, root['lower_bound'])
        if best is None or upper < best['upper']:
            best = dict(q0=q0.copy(), branch_q=branch_q.copy(), upper=upper, root=root,
                        tail_q=tail_q.copy(), soc6=soc6, soc24=terminal,
                        first24_objective=first24, branch_ordinary=ordinary,
                        branch_adjustment=adjustment, branch_emergency=emergency,
                        branch_tail=tail_replay)
        relative = max(0., (best['upper']-global_lower)/max(abs(best['upper']), 1e-12))
        iterations.append(dict(iteration=iteration, upper=upper, best_upper=best['upper'],
                               lower=global_lower, gap=relative, root_gap=root['gap'],
                               root_seconds=root['seconds'], root_relaxation_seconds=root['relaxation_seconds'],
                               root_replay_error=root['replay_max_error']))
        if relative <= TARGET_GAP+1e-9:
            break
        added = False
        for k in range(3):
            if all(abs(nxt[k]['E']-z['E']) > 1e-5 for z in points[k]):
                points[k].append(nxt[k]); added = True
        if not added:
            break
    assert np.max(np.abs(best['soc6']-best['soc6'][0])) < 1e-5
    stats = dict(incumbent=best['upper'], lower_bound=global_lower, gap=relative,
                 seconds=time.perf_counter()-start, iterations=iterations, root=best['root'],
                 grid_points=[len(z) for z in points], unique_tail_evaluations=len(cache),
                 horizon_steps=288, first24_objective=best['first24_objective'],
                 tail24_objective=best['upper']-best['first24_objective'],
                 target_gap=TARGET_GAP, target_met=bool(relative <= TARGET_GAP+1e-9),
                 strict_target_gap=.0001, strict_target_met=bool(relative <= .0001+1e-9),
                 root_time_limit_seconds=ROOT_LIMIT, max_continuation_iterations=MAX_ITER,
                 information_branches=3, residual_branches_per_information=1,
                 contract_mode='variable_midnight_original' if reference is None else 'fixed_original_reference',
                 returned_postfix_role='original_contract' if reference is None else 'weighted_median_diagnostic',
                 committed_reference_steps=0 if reference is None else int(np.isfinite(reference).sum()))
    detail = {key: best[key] for key in ('branch_q', 'tail_q', 'soc6', 'soc24',
               'branch_ordinary', 'branch_adjustment', 'branch_emergency', 'branch_tail')}
    detail['weights'] = weights.copy()
    detail['reference'] = np.full(144, np.nan) if reference is None else reference.copy()
    detail['provisional_q'] = best['tail_q'].copy()
    detail['terminal'] = best['soc24'].copy()
    return best['q0'], stats, detail
