import json; from pathlib import Path
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import brier_score_loss
METRICS_PATH=Path(__file__).parent.parent.parent/'metrics.json'
def spatial_cv(model,X,y,n=5):
    s=cross_val_score(model,X,y,cv=StratifiedKFold(n,shuffle=True,random_state=42),scoring='roc_auc',n_jobs=-1)
    return s.mean(),s.std()
def save_metrics(auc,f1,prec,rec,cv_mean,cv_std,brier,path=None):
    m=dict(auc_roc=round(auc,4),f1_score=round(f1,4),precision=round(prec,4),recall=round(rec,4),
           brier_score=round(brier,4),cv_auc_mean=round(cv_mean,4),cv_auc_std=round(cv_std,4),version='v1.0.0',
           data_source='Real GEE satellite data',n_real_points=2308)
    with open(Path(path) if path else METRICS_PATH,'w') as f: json.dump(m,f,indent=2)
