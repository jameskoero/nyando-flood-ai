"""Export the registered Phase C model (docs/PHASE_C_PROTOCOL.md, Section 15) to models/nyando_logcon_<sha256 prefix>.onnx.

Run from the repository root in an environment with onnx installed (requirements-onnx.txt):
    python scripts/export_registered.py          # prints the file name, SHA-256 and size; writes nothing
    python scripts/export_registered.py --write  # also writes the file into models/"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.constrained import build_constrained_logistic
from src.models.cv import FEATURE_SETS, FLAGS_PATH, LABEL, load_layer_flags, load_training_frame, restrict_to_mappable
from src.models.export_onnx import to_onnx

REGISTRATION_RUN = "54afe7e812fd4c4985280283d52f3793"
STEM = "nyando_logcon_"
CLAIM_LIMIT = ("Scores rank locations inside the areas GFM can map. They are not flood probabilities (the sample is case-control) "
               "and say nothing about locations inside the GFM exclusion mask or about floods GFM cannot detect.")


def build():
    df, data_hash = load_training_frame()
    mapp = restrict_to_mappable(df, load_layer_flags(df))
    feats = FEATURE_SETS["six"]
    model = build_constrained_logistic(feats).fit(mapp[feats], mapp[LABEL].to_numpy())
    meta = {"model": "logistic:con", "training_frame": "mappable", "rows": len(mapp), "features": ",".join(feats),
            "training_data_sha256": data_hash, "layer_flags_sha256": hashlib.sha256(Path(FLAGS_PATH).read_bytes()).hexdigest(),
            "protocol": "docs/PHASE_C_PROTOCOL.md Section 15", "registration_run": REGISTRATION_RUN,
            "algorithm": "logistic regression, L2 (C=1), standardized features; rainfall_3day >= 0 and elevation, distance_river, slope <= 0",
            "input_contract": "float64 [n,1] for each numeric feature (NaN = missing clay_percent); int64 [n,1] land_cover; output probabilities [n,2] = [no flood, flood]",
            "claim_limit": CLAIM_LIMIT}
    return to_onnx(model, feats, meta), meta


def main():
    blob, _ = build()
    digest = hashlib.sha256(blob).hexdigest()
    name = "%s%s.onnx" % (STEM, digest[:12])
    print(name, digest, len(blob))
    if "--write" in sys.argv:
        (ROOT / "models" / name).write_bytes(blob)
        print("wrote models/" + name)


if __name__ == "__main__":
    main()
