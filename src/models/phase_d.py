"""Phase D rules as code (docs/PHASE_D_PROTOCOL.md, register row D46). Pure Python: no torch, no data, no network. Block B stays sealed until docs/PHASE_D_FREEZE.json is merged."""
COMPARATOR = "models/nyando_hgbcon_5ae81ad8b030.onnx"
N_NUMERIC = 5  # elevation, slope, rainfall_3day, distance_river, clay_percent
LAND_COVER_CLASSES = (10, 20, 30, 40, 50, 60, 80, 90)
N_INPUTS = N_NUMERIC + 1 + len(LAND_COVER_CLASSES)  # plus the blank-clay indicator
CONSTRAINTS = {"rainfall_3day": 1, "elevation": -1, "distance_river": -1, "slope": -1}
HIDDEN = ((32, 16), (64, 32))
LAMBDAS = (0.0, 1.0, 10.0)
GRID = tuple({"hidden": h, "lam": lam} for h in HIDDEN for lam in LAMBDAS)  # simplest first: smaller network, then smaller penalty weight
FIXED = {"activation": "tanh", "optimizer": "adam", "lr": 0.001, "weight_decay": 0.0001, "batch_size": 256, "epochs": 100, "seed": 42, "dtype": "float32", "device": "cpu"}
SEEDS_REPORTED = (42, 43, 44, 45, 46)
INNER_FOLDS = 4
MIN_BLOCK_B_EVENTS = 15
PARITY_TOLERANCE = 1e-5
PROMOTED = "PHASE D PROMOTED"
NOT_PROMOTED = "PHASE D NOT PROMOTED - hgb:con RETAINED"


def param_count(hidden, n_in=N_INPUTS):
    dims = [n_in] + list(hidden) + [1]
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))


def estimable(n_scorable_events):
    return n_scorable_events >= MIN_BLOCK_B_EVENTS


def wins(stat):
    """A beats B: the paired per-event bootstrap interval of A minus B has a lower end above 0 and A has the higher mean."""
    return stat["mean_diff"] > 0 and stat["ci_low"] > 0


def gates(n_events, vs_hgbcon_mappable, vs_hgbcon_full, vs_elevation_mappable, no_floor_mean_diff):
    ok = estimable(n_events)
    return {"G1": bool(ok and wins(vs_hgbcon_mappable) and wins(vs_hgbcon_full)), "G2": bool(ok and wins(vs_elevation_mappable)), "G3": bool(ok and no_floor_mean_diff > 0)}


def export_gates(max_abs_diff, per_event_auc_diff, violations, sha_ok):
    return {"X1": max_abs_diff <= PARITY_TOLERANCE, "X2": per_event_auc_diff <= PARITY_TOLERANCE, "X3": all(int(v) == 0 for v in violations.values()), "X4": bool(sha_ok)}


def decision(statistical, export=None):
    """One outcome. None while the statistical gates pass and the export gates are not yet computed."""
    if not all(statistical.values()):
        return NOT_PROMOTED
    if export is None:
        return None
    return PROMOTED if all(export.values()) else NOT_PROMOTED


def block_b_allowed(freeze_exists, block_b_paths):
    """Block B may be built only after the freeze file is merged."""
    return bool(freeze_exists) or not list(block_b_paths)


# Stage 2 procedure (docs/PHASE_D_PROTOCOL.md Section 9, r2)
MIN_CLASS_N = 30      # rows of each class an event needs to be scorable (src/models/cv.py)
N_BOOT = 10000
ALPHA = 0.025         # 97.5% paired per-event bootstrap interval
REPRODUCE = {"hgb:con": (0.943, 0.001), "logistic:con": (0.937239, 0.0005)}  # mean per-event AUC on the existing events, mappable frame: README and Phase C record; reported, never a gate


def _key(p):
    return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in p.items()))


def modal_point(chosen, grid):
    """The grid point chosen by most outer folds; ties go to the earlier (simpler) grid point."""
    keys = [_key(c) for c in chosen]
    return max(grid, key=lambda g: (keys.count(_key(g)), -list(grid).index(g)))
