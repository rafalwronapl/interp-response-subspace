"""Paired lexical-block inference; all contrasts oriented positive for support.

No complete-case deletion, epsilons, alternative tests after degeneracy, or
substitution of inconclusive hypotheses out of the three-slot Holm family.
"""
import numpy as np


def holm(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Exactly three valid family p values required')
    order = np.argsort(p, kind='stable')
    out = np.empty(3)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order]*np.arange(3, 0, -1)))
    return out.tolist()


def inconclusive(reason):
    return dict(status='inconclusive', reason=reason, p_for_holm=1.0)


def mean_contrast(x, draws=20000, seed=20261101, alpha=.01):
    """One-sided centered studentized paired-block bootstrap and bootstrap-t CI."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        return inconclusive('missing/nonfinite primary cell or fewer than 3 independent blocks')
    n = len(x)
    est = x.mean()
    se = x.std(ddof=1)/np.sqrt(n)
    if not np.isfinite(se) or se <= 0:
        return inconclusive('zero/nonfinite observed variance')
    rng = np.random.Generator(np.random.PCG64(seed))
    ts = []
    for start in range(0, draws, 1000):
        z = x[rng.integers(n, size=(min(1000, draws-start), n))]
        sz = z.std(axis=1, ddof=1)/np.sqrt(n)
        if np.any(~np.isfinite(sz)) or np.any(sz <= 0):
            return inconclusive('degenerate bootstrap draw')
        ts.extend((z.mean(axis=1)-est)/sz)
    ts = np.asarray(ts)
    qlo, qhi = np.quantile(ts, [alpha/2, 1-alpha/2])
    return dict(status='defined', estimate=float(est), se=float(se), n=n,
                p_for_holm=float((1+np.count_nonzero(ts >= est/se))/(draws+1)),
                ci=[float(est-qhi*se), float(est-qlo*se)], draws=draws, seed=seed)


def ratio_contrast(energies, draws=20000, seed=20261101, alpha=.01):
    """log(E_small/S_small)-log(E_one/S_one); joint energy resampling.

    Columns: E_small,S_small,E_one,S_one. Delta-method influence SE is
    recomputed in every bootstrap draw. Nonpositive total energies fail closed.
    """
    a = np.asarray(energies, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4 or len(a) < 3 or not np.isfinite(a).all() or np.any(a < 0):
        return inconclusive('invalid/missing energies')
    signs = np.array([1., -1., -1., 1.])

    def calc(z):
        m = z.mean(axis=-2)
        if np.any(m <= 0):
            raise ValueError('zero pooled energy')
        theta = (np.log(m)*signs).sum(axis=-1)
        influence = ((z/m[..., None, :]-1)*signs).sum(axis=-1)
        se = influence.std(axis=-1, ddof=1)/np.sqrt(len(a))
        if np.any(~np.isfinite(se)) or np.any(se <= 0):
            raise ValueError('zero/nonfinite influence variance')
        return theta, se
    try:
        est, se = calc(a)
        rng = np.random.Generator(np.random.PCG64(seed))
        ts = []
        for start in range(0, draws, 500):
            z = a[rng.integers(len(a), size=(min(500, draws-start), len(a)))]
            t, s = calc(z)
            ts.extend((t-est)/s)
    except ValueError as exc:
        return inconclusive(str(exc))
    ts = np.asarray(ts)
    qlo, qhi = np.quantile(ts, [alpha/2, 1-alpha/2])
    return dict(status='defined', estimate=float(est), se=float(se), n=len(a),
                p_for_holm=float((1+np.count_nonzero(ts >= est/se))/(draws+1)),
                ci=[float(est-qhi*se), float(est-qlo*se)], draws=draws, seed=seed)


def family(components):
    """Three hypotheses, each containing all required IUT components."""
    if len(components) != 3 or any(not x for x in components):
        raise ValueError('Three nonempty hypotheses required')
    p = [max(c['p_for_holm'] for c in slot) for slot in components]
    adjusted = holm(p)
    return [dict(components=slot, p=p[i], p_holm=adjusted[i],
                 status='inconclusive' if any(c['status'] != 'defined' for c in slot)
                 else ('supported' if adjusted[i] <= .05 else 'not_supported'))
            for i,slot in enumerate(components)]


def crossed_summary(values, draws=20000, seed=20261101):
    """Separate descriptive FIT-cohort x shared-TEST block bootstrap.

    Every replicate independently resamples rows (FIT realizations) and columns
    (TEST blocks); never treats their Cartesian product as independent rows.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 3 or not np.isfinite(x).all():
        return inconclusive('incomplete crossed FIT x TEST matrix')
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = np.empty(draws)
    for i in range(draws):
        boot[i] = x[np.ix_(rng.integers(x.shape[0], size=x.shape[0]),
                          rng.integers(x.shape[1], size=x.shape[1]))].mean()
    return dict(status='descriptive', estimate=float(x.mean()), per_fit=x.mean(axis=1).tolist(),
                fit_sd=float(x.mean(axis=1).std(ddof=1)), ci=np.quantile(boot, [.005,.995]).tolist(),
                cohorts=x.shape[0], blocks=x.shape[1], draws=draws,construction_confidence=.99,
                population_coverage_goal=.95,coverage_established=False)
