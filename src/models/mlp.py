"""Phase D model (docs/PHASE_D_PROTOCOL.md Section 1): a small fully connected tanh network, one sigmoid output, trained with binary cross-entropy plus the penalty weight times a soft
constraint penalty. The penalty is the mean over the real rows of a batch of the summed positive part of (minus the required sign times dP/dx), with dP/dx taken by autograd in
standardised units (a positive rescaling keeps the sign). No synthetic rows are used. Preprocessing is inside the estimator (mlp_pre) so the exporter can bake it into the graph.
torch is imported inside the functions: importing this module needs only numpy and scikit-learn. Named a physics-constrained MLP; there is no PDE loss."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.mlp_pre import NUMERIC, Preprocessor
from src.models.phase_d import CONSTRAINTS, FIXED, N_INPUTS

FEATURES = NUMERIC + ("land_cover",)
SIGNS = tuple(CONSTRAINTS.get(c, 0) for c in NUMERIC)


def build_net(hidden, n_in=N_INPUTS):
    import torch
    dims, layers = [n_in] + list(hidden), []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], 1))


def constraint_penalty(net, xb, signs=SIGNS):
    """xb must require grad. Returns the mean over its rows of the summed positive part of (minus sign times dP/dx) at the constrained inputs, P = sigmoid(net(x))."""
    import torch
    p = torch.sigmoid(net(xb).squeeze(1))
    g = torch.autograd.grad(p.sum(), xb, create_graph=True)[0][:, : len(signs)]
    return torch.relu(-torch.tensor(signs, dtype=xb.dtype) * g).sum(dim=1).mean()


class PhysicsMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(32, 16), lam=1.0, seed=FIXED["seed"], epochs=FIXED["epochs"], batch_size=FIXED["batch_size"], lr=FIXED["lr"], weight_decay=FIXED["weight_decay"]):
        self.hidden, self.lam, self.seed, self.epochs = hidden, lam, seed, epochs
        self.batch_size, self.lr, self.weight_decay = batch_size, lr, weight_decay

    def fit(self, X, y):
        import torch
        self.classes_ = np.array([0, 1])
        self.pre_ = Preprocessor().fit(X)
        Xt, yt = torch.tensor(self.pre_.transform(X)), torch.tensor(np.asarray(y, dtype=np.float32))
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        net = build_net(self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), self.batch_size):
                ix = perm[i: i + self.batch_size]
                xb = Xt[ix].clone().requires_grad_(self.lam > 0)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xb).squeeze(1), yt[ix])
                if self.lam > 0:
                    loss = loss + self.lam * constraint_penalty(net, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net_ = net.eval()
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(self.pre_.transform(X))).squeeze(1)).numpy().astype(np.float64)
        return np.column_stack([1.0 - p, p])
