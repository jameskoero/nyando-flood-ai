"""
shap_plots.py — SHAP explainability plots for Nyando Flood AI.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_shap_summary(model, X, feature_names=None,
                      filename="shap_summary.png"):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    feat_names  = feature_names or list(X.columns)
    shap.summary_plot(shap_values, X,
                      feature_names=feat_names, show=False)
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] SHAP summary saved -> {path}")
    return path


def plot_roc_curve(models_dict: dict, X_test, y_test,
                   filename="roc_curve.png"):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(7,6))
    fig.patch.set_facecolor("#0A1628")
    ax.set_facecolor("#0D1E38")
    colors = ["#C9A84C","#2ECC71","#3498DB","#E74C3C"]
    for (name, model), color in zip(models_dict.items(), colors):
        probs = model.predict_proba(X_test)[:,1]
        auc   = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=color, lw=2)
    ax.plot([0,1],[0,1],"w--",lw=1,alpha=0.4)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curve — Nyando Flood Models",
                 color="#C9A84C", fontsize=13)
    ax.legend(facecolor="#0A1628", labelcolor="white")
    ax.tick_params(colors="white")
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] ROC curve saved -> {path}")
    return path
