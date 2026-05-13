import numpy as np
FEATURES=['elevation','slope','rainfall_3day','distance_river','clay_percent','land_cover']; TARGET='flooded'
def clean_features(df):
    return df.dropna(subset=FEATURES+[TARGET]).query('elevation>0 and rainfall_3day>=0').reset_index(drop=True)
def apply_smote(X,y):
    min_idx=np.where(y==1)[0]; maj_idx=np.where(y==0)[0]; n=len(maj_idx)-len(min_idx)
    s=[X[i:=np.random.choice(min_idx)]*a+X[j:=np.random.choice(min_idx)]*(1-a) for a in np.random.uniform(0.2,0.8,n) for _ in [1]]
    return np.vstack([X,np.array(s[:n])]),np.concatenate([y,np.ones(n,int)])
