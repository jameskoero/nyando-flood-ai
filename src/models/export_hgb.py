"""ONNX export of hgb:con (docs/PROMOTION_PROTOCOL.md, export gates E1 to E4). A HistGradientBoostingClassifier with a categorical land_cover is written as one
ai.onnx.ml TreeEnsemble (opset 5) in double precision, with the production input contract of backend/registered.py: six [N, 1] inputs (double, land_cover int64)
and a double [N, 2] output. skl2onnx 1.20.0 is not used: it ignores scikit-learn's internal column order and has no category splits."""
import json

import numpy as np
import onnx
import sklearn
from onnx import TensorProto as T
from onnx import helper, numpy_helper

FEATURES = ("elevation", "slope", "rainfall_3day", "distance_river", "clay_percent", "land_cover")
CATEGORICAL, OUTPUT = "land_cover", "probabilities"
OPSET, ML_OPSET, IR_VERSION = 13, 5, 10
NODE_FIELDS = ("value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf", "is_categorical", "bitset_idx")


def _guard(model):
    """Fail loudly if scikit-learn's private representation is not the one this exporter was written and tested against."""
    if not sklearn.__version__.startswith("1.6."):
        raise RuntimeError("written for scikit-learn 1.6.x, found %s" % sklearn.__version__)
    if list(model.feature_names_in_) != list(FEATURES):
        raise ValueError("input order %s, expected %s" % (list(model.feature_names_in_), list(FEATURES)))
    if model.n_trees_per_iteration_ != 1 or len(model.classes_) != 2:
        raise ValueError("a binary classifier is expected")
    cat = getattr(model, "is_categorical_", None)
    if cat is None or list(np.flatnonzero(cat)) != [FEATURES.index(CATEGORICAL)]:
        raise ValueError("land_cover must be the only categorical feature")
    missing = set(NODE_FIELDS) - set(model._predictors[0][0].nodes.dtype.names)
    if missing:
        raise RuntimeError("the tree node record lacks %s" % sorted(missing))


def internal_order(model):
    """Column order inside the fitted model: the preprocessor's transformers in order, each over its own columns."""
    names, order = list(model.feature_names_in_), []
    for _, tr, mask in model._preprocessor.transformers_:
        if isinstance(tr, str) and tr == "drop":
            continue
        order += [f for f, m in zip(names, mask) if m]
    if sorted(order) != sorted(names):
        raise RuntimeError("the internal column order does not cover every input: %s" % order)
    return order


def categories(model):
    cats = [getattr(tr, "categories_", None) for _, tr, _ in model._preprocessor.transformers_]
    cats = [c for c in cats if c is not None]
    if len(cats) != 1 or len(cats[0]) != 1:
        raise RuntimeError("exactly one categorical encoder with one column is expected")
    return [float(c) for c in cats[0][0]]


def tree_node(model, cats):
    """Every tree as TreeEnsemble-5 arrays. Numeric split: value <= threshold goes to the true branch. Category split: membership in the
    classes whose codes are set in the node's bitset goes to the true branch. A blank follows missing_go_to_left at every node."""
    n = {k: [] for k in ("feat", "mode", "split", "true", "tleaf", "false", "fleaf", "miss")}
    leaves, members, roots = [], [], []
    for stage in model._predictors:
        p = stage[0]
        nd = p.nodes
        def visit(i):
            if nd[i]["is_leaf"]:
                leaves.append(float(nd[i]["value"]))
                return len(leaves) - 1, 1
            k = len(n["feat"])
            for v in n.values():
                v.append(0)
            n["feat"][k], n["miss"][k] = int(nd[i]["feature_idx"]), int(nd[i]["missing_go_to_left"])
            if nd[i]["is_categorical"]:
                bits = p.raw_left_cat_bitsets[int(nd[i]["bitset_idx"])]
                members.append([cats[c] for c in range(len(cats)) if (int(bits[c // 32]) >> (c % 32)) & 1])
                n["mode"][k], n["split"][k] = 6, 0.0
            else:
                n["mode"][k], n["split"][k] = 0, float(nd[i]["num_threshold"])
            n["true"][k], n["tleaf"][k] = visit(int(nd[i]["left"]))
            n["false"][k], n["fleaf"][k] = visit(int(nd[i]["right"]))
            return k, 0
        k, leaf = visit(0)
        if leaf:   # a tree that is only a leaf: one split whose two branches reach that leaf
            r = len(n["feat"])
            for key, v in (("feat", 0), ("mode", 0), ("split", 0.0), ("true", k), ("tleaf", 1), ("false", k), ("fleaf", 1), ("miss", 0)):
                n[key].append(v)
            k = r
        roots.append(k)
    flat = []
    for j, s in enumerate(members):
        flat += ([np.nan] if j else []) + s
    attrs = dict(tree_roots=roots, nodes_featureids=n["feat"], nodes_truenodeids=n["true"], nodes_trueleafs=n["tleaf"], nodes_falsenodeids=n["false"],
                 nodes_falseleafs=n["fleaf"], nodes_missing_value_tracks_true=n["miss"], leaf_targetids=[0] * len(leaves), n_targets=1, aggregate_function=1, post_transform=0,
                 nodes_modes=numpy_helper.from_array(np.array(n["mode"], np.uint8)), nodes_splits=numpy_helper.from_array(np.array(n["split"], np.float64)),
                 leaf_weights=numpy_helper.from_array(np.array(leaves, np.float64)))
    if flat:
        attrs["membership_values"] = numpy_helper.from_array(np.array(flat, np.float64))
    return helper.make_node("TreeEnsemble", ["X"], ["raw"], domain="ai.onnx.ml", **attrs), {"trees": len(roots), "nodes": len(n["feat"]), "leaves": len(leaves), "member_nodes": len(members)}


def to_onnx_hgb(model, metadata=None):
    _guard(model)
    order, cats = internal_order(model), categories(model)
    if order[0] != CATEGORICAL:
        raise RuntimeError("land_cover is expected first in the internal order, found %s" % order)
    tree, info = tree_node(model, cats)
    nodes = [helper.make_node("Cast", [CATEGORICAL], ["lc_double"], to=T.DOUBLE),
             helper.make_node("Concat", ["lc_double" if f == CATEGORICAL else f for f in order], ["X"], axis=1), tree,
             helper.make_node("Add", ["raw", "baseline"], ["logit"]), helper.make_node("Sigmoid", ["logit"], ["p1"]),
             helper.make_node("Sub", ["one", "p1"], ["p0"]), helper.make_node("Concat", ["p0", "p1"], [OUTPUT], axis=1)]
    inits = [numpy_helper.from_array(np.array([[float(np.ravel(model._baseline_prediction)[0])]], np.float64), "baseline"), numpy_helper.from_array(np.array([[1.0]], np.float64), "one")]
    inputs = [helper.make_tensor_value_info(f, T.INT64 if f == CATEGORICAL else T.DOUBLE, [None, 1]) for f in FEATURES]
    graph = helper.make_graph(nodes, "nyando_hgbcon", inputs, [helper.make_tensor_value_info(OUTPUT, T.DOUBLE, [None, 2])], initializer=inits)
    out = helper.make_model(graph, producer_name="nyando-flood-ai", opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid("ai.onnx.ml", ML_OPSET)])
    out.ir_version = IR_VERSION
    helper.set_model_props(out, {"model": "hgb:con", "land_cover_classes": ",".join(str(int(c)) for c in cats), **{k: (v if isinstance(v, str) else json.dumps(v)) for k, v in (metadata or {}).items()}})
    onnx.checker.check_model(out)
    return out, info
