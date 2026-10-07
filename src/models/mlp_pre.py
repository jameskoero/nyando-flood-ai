"""Phase D model input (docs/PHASE_D_PROTOCOL.md Section 1): the fitted preprocessing of the physics-constrained MLP. numpy only, no torch, so CI can test it.
clay_percent blanks are filled with the training median of its observed values; the five numeric columns are then standardised with the mean and the population
standard deviation (ddof 0) of the filled training column; one indicator is 1 where clay_percent was blank; land_cover is one-hot over the eight classes and an
unseen class gives all zeros. Only clay_percent may be blank: any other blank is an error, never silently filled."""
import numpy as np

from src.models.phase_d import LAND_COVER_CLASSES, N_INPUTS

NUMERIC = ("elevation", "slope", "rainfall_3day", "distance_river", "clay_percent")
BLANK = NUMERIC.index("clay_percent")


class Preprocessor:
    def _numeric(self, X):
        v = np.column_stack([np.asarray(X[c], dtype=float) for c in NUMERIC])
        bad = [c for j, c in enumerate(NUMERIC) if j != BLANK and np.isnan(v[:, j]).any()]
        if bad:
            raise ValueError("blank values in %s: only clay_percent may be blank" % bad)
        return v

    def fit(self, X):
        v = self._numeric(X)
        if np.isnan(v[:, BLANK]).all():
            raise ValueError("clay_percent has no observed value to take a median from")
        self.median_ = float(np.nanmedian(v[:, BLANK]))
        v[:, BLANK] = np.where(np.isnan(v[:, BLANK]), self.median_, v[:, BLANK])
        self.mean_, sd = v.mean(axis=0), v.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X):
        v = self._numeric(X)
        blank = np.isnan(v[:, BLANK])
        v[:, BLANK] = np.where(blank, self.median_, v[:, BLANK])
        onehot = np.asarray(X["land_cover"]).reshape(-1, 1) == np.array(LAND_COVER_CLASSES).reshape(1, -1)
        out = np.column_stack([(v - self.mean_) / self.scale_, blank.astype(float), onehot.astype(float)]).astype(np.float32)
        if out.shape[1] != N_INPUTS:
            raise ValueError("expected %d inputs, built %d" % (N_INPUTS, out.shape[1]))
        return out
