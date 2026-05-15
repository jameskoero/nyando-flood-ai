import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import numpy as np; from pathlib import Path
FIGS=Path(__file__).parent.parent.parent/'reports/figures'
def plot_importance(model,feature_names,save=True):
    FIGS.mkdir(parents=True,exist_ok=True); imp=model.feature_importances_; idx=np.argsort(imp)
    fig,ax=plt.subplots(figsize=(9,5)); fig.patch.set_facecolor('#0A1628'); ax.set_facecolor('#0E1E35')
    ax.barh([feature_names[i] for i in idx],[imp[i] for i in idx],color='#C9A84C')
    ax.set_title('Feature Importance — Real GEE Data',color='#C9A84C',fontweight='bold')
    plt.tight_layout()
    if save: plt.savefig(FIGS/'shap_summary.png',dpi=150,bbox_inches='tight',facecolor='#0A1628')
    plt.close()
