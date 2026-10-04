"""Exact ONNX export of the Phase C logistic pipelines (docs/PHASE_C_PROTOCOL.md, Section 15).

skl2onnx 1.20.0 has no converter for sklearn.impute.MissingIndicator (conversion failed on 2026-10-02), so the pipeline built by
src/models/baseline.py is written out as an explicit ONNX graph from its fitted parameters: median imputation and standardization of
the numeric features, a missing-value indicator for clay_percent, a one-hot land_cover, then a linear score and a sigmoid. The exporter
checks the structure it understands and refuses anything else. Exporting needs onnx; scoring needs onnxruntime."""
import numpy as np

OPSET = 13
IR_VERSION = 8
OUTPUT = "probabilities"
CATEGORICAL_INPUT = "land_cover"
INDICATOR_COLUMN = "clay_percent"


class UnsupportedPipeline(ValueError):
    """The pipeline does not have the structure this exporter reproduces exactly."""


def _need(ok, message):
    if not ok:
        raise UnsupportedPipeline(message)


def _is_nan(value):
    return isinstance(value, float) and bool(np.isnan(value))


def read_pipeline(pipe, features):
    """Validate the pipeline structure and return its fitted parameters as plain numpy values."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    _need(list(pipe.named_steps) == ["prep", "clf"], "expected the steps prep and clf")
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    _need(isinstance(prep, ColumnTransformer), "prep is not a ColumnTransformer")
    parts = {name: (tr, list(cols)) for name, tr, cols in prep.transformers_ if name != "remainder"}
    _need(list(parts) == ["num", "blank", "cat"], "expected the transformers num, blank and cat in that order")

    num_pipe, num_cols = parts["num"]
    _need([type(s).__name__ for _, s in num_pipe.steps] == ["SimpleImputer", "StandardScaler"],
          "num must be a SimpleImputer followed by a StandardScaler")
    imp, scaler = num_pipe.steps[0][1], num_pipe.steps[1][1]
    _need(isinstance(imp, SimpleImputer) and imp.strategy == "median" and _is_nan(imp.missing_values) and not imp.add_indicator,
          "the imputer must be a plain median imputer for NaN")
    _need(isinstance(scaler, StandardScaler) and scaler.with_mean and scaler.with_std, "the scaler must centre and scale")

    ind, ind_cols = parts["blank"]
    _need(isinstance(ind, MissingIndicator) and ind.features == "all" and _is_nan(ind.missing_values),
          "blank must be MissingIndicator(features='all') for NaN")
    _need(ind_cols == [INDICATOR_COLUMN] and INDICATOR_COLUMN in num_cols, "the indicator must be for clay_percent, which must also be numeric")

    enc, cat_cols = parts["cat"]
    _need(isinstance(enc, OneHotEncoder) and enc.handle_unknown == "ignore" and enc.drop is None
          and enc.min_frequency is None and enc.max_categories is None, "cat must be a plain OneHotEncoder that ignores unknown values")
    _need(cat_cols == [CATEGORICAL_INPUT] and len(enc.categories_) == 1, "land_cover must be the only categorical feature")
    cats = [int(c) for c in enc.categories_[0]]
    _need(sorted(features) == sorted(num_cols + [CATEGORICAL_INPUT]), "the features do not match the pipeline columns")

    _need(np.asarray(getattr(clf, "classes_", [])).tolist() == [0, 1], "a binary 0/1 classifier is required")
    w = np.asarray(clf.coef_, dtype=float).reshape(-1)
    _need(w.shape[0] == len(num_cols) + 1 + len(cats), "the coefficient count does not match the design matrix")
    return {"num_cols": num_cols, "median": np.asarray(imp.statistics_, dtype=float), "mean": np.asarray(scaler.mean_, dtype=float),
            "scale": np.asarray(scaler.scale_, dtype=float), "cats": cats, "weights": w, "intercept": float(np.ravel(clf.intercept_)[0])}


def to_onnx(pipe, features, metadata=None):
    """Serialized ONNX model (bytes) of a fitted pipeline built by build_logistic or build_constrained_logistic."""
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper

    p = read_pipeline(pipe, features)
    nodes, inits = [], []

    def const(name, array):
        inits.append(numpy_helper.from_array(np.asarray(array), name=name))
        return name

    def node(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    scaled, nan_of = [], {}
    for i, f in enumerate(p["num_cols"]):
        nan_of[f] = node("IsNaN", [f], "isnan_" + f)
        imputed = node("Where", [nan_of[f], const("median_" + f, [p["median"][i]]), f], "imputed_" + f)
        centred = node("Sub", [imputed, const("mean_" + f, [p["mean"][i]])], "centred_" + f)
        scaled.append(node("Div", [centred, const("scale_" + f, [p["scale"][i]])], "scaled_" + f))
    columns = scaled + [node("Cast", [nan_of[INDICATOR_COLUMN]], "indicator_" + INDICATOR_COLUMN, to=int(T.DOUBLE))]
    for c in p["cats"]:
        eq = node("Equal", [CATEGORICAL_INPUT, const("category_%d" % c, np.array([c], dtype=np.int64))], "is_%d" % c)
        columns.append(node("Cast", [eq], "onehot_%d" % c, to=int(T.DOUBLE)))
    design = node("Concat", columns, "design", axis=1)
    linear = node("MatMul", [design, const("weights", p["weights"].reshape(-1, 1))], "linear")
    logit = node("Add", [linear, const("intercept", [p["intercept"]])], "logit")
    p1 = node("Sigmoid", [logit], "p1")
    p0 = node("Sub", [const("one", [1.0]), p1], "p0")
    node("Concat", [p0, p1], OUTPUT, axis=1)

    inputs = [helper.make_tensor_value_info(f, T.INT64 if f == CATEGORICAL_INPUT else T.DOUBLE, [None, 1]) for f in features]
    graph = helper.make_graph(nodes, "nyando_logistic", inputs, [helper.make_tensor_value_info(OUTPUT, T.DOUBLE, [None, 2])], initializer=inits)
    model = helper.make_model(graph, producer_name="nyando-flood-ai", opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    for key, value in sorted((metadata or {}).items()):
        entry = model.metadata_props.add()
        entry.key, entry.value = str(key), str(value)
    onnx.checker.check_model(model)
    try:
        return model.SerializeToString(deterministic=True)
    except TypeError:
        return model.SerializeToString()


class OnnxLogistic:
    """onnxruntime session with the same input contract as the pipeline: predict_proba(frame) gives an (n, 2) float64 array."""

    def __init__(self, blob, features):
        import onnxruntime as ort
        self.features = list(features)
        self.session = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])

    def predict_proba(self, frame):
        feed = {f: frame[[f]].to_numpy(dtype=np.int64 if f == CATEGORICAL_INPUT else np.float64) for f in self.features}
        return self.session.run([OUTPUT], feed)[0]
