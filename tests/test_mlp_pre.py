"""Phase D Stage 1 (register row D46): the MLP's preprocessing on the real training rows, the grid equal to the protocol's, torch kept in requirements-train.txt, and the README tick following the files."""
import re
from pathlib import Path
import numpy as np
import pytest
from src.models import phase_d as D
from src.models.boosters import expand
from src.models.cv import FLAGS_PATH, load_layer_flags, load_training_frame, restrict_to_mappable
from src.models.mlp_pre import NUMERIC, Preprocessor

ROOT = Path(__file__).resolve().parent.parent
DF, _ = load_training_frame(ROOT)
FRAME = restrict_to_mappable(DF, load_layer_flags(DF, FLAGS_PATH)).reset_index(drop=True)
K = len(NUMERIC)


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_inputs_have_the_protocols_shape_and_the_blank_indicator_follows_clay():
    Z = Preprocessor().fit(FRAME).transform(FRAME)
    assert Z.shape == (len(FRAME), D.N_INPUTS) and Z.dtype == np.float32
    assert np.array_equal(Z[:, K] == 1, FRAME["clay_percent"].isna().to_numpy())


def test_training_columns_are_standardised_and_the_fill_is_the_observed_median():
    pre = Preprocessor().fit(FRAME)
    Z = pre.transform(FRAME)[:, :K]
    sd = Z.std(axis=0)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-4) and np.allclose(sd[sd > 0], 1, atol=1e-4)
    assert pre.median_ == pytest.approx(float(FRAME["clay_percent"].median()))


def test_land_cover_is_one_hot_and_an_unseen_class_gives_zeros():
    pre = Preprocessor().fit(FRAME)
    hot = pre.transform(FRAME)[:, K + 1:]
    rows = hot.sum(axis=1) == 1
    assert hot.shape[1] == len(D.LAND_COVER_CLASSES) and hot.sum(axis=1).max() <= 1 and rows.any()
    assert (np.array(D.LAND_COVER_CLASSES)[hot[rows].argmax(axis=1)] == FRAME["land_cover"].to_numpy()[rows]).all()
    odd = FRAME.head(3).copy()
    odd["land_cover"] = 999
    assert pre.transform(odd)[:, K + 1:].sum() == 0


def test_transform_uses_the_training_statistics_and_leaves_its_input_alone():
    pre = Preprocessor().fit(FRAME)
    head = FRAME.head(50)
    before = head.copy()
    assert np.array_equal(pre.transform(head), pre.transform(FRAME)[:50]) and head.equals(before)


def test_a_blank_outside_clay_is_an_error_not_a_fill():
    bad = FRAME.head(5).copy()
    bad.loc[bad.index[0], "elevation"] = np.nan
    with pytest.raises(ValueError, match="only clay_percent may be blank"):
        Preprocessor().fit(bad)


def test_the_grid_the_search_expands_equals_the_protocol_grid():
    assert expand({"hidden": D.HIDDEN, "lam": D.LAMBDAS}) == list(D.GRID)


def test_torch_is_pinned_only_in_requirements_train_and_the_readme_tick_follows_the_files():
    assert re.search(r"^torch==\S+$", (ROOT / "requirements-train.txt").read_text(encoding="utf-8"), re.M)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ("- [x] **D. Stage 1:**" in readme) == (ROOT / "src" / "models" / "mlp.py").exists()
