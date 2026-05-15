import numpy as np
FEATURES=['elevation','slope','rainfall_3day','distance_river','clay_percent','land_cover']
def clean(df): return df.dropna(subset=FEATURES+['flooded']).query('elevation>0 and rainfall_3day>=0').reset_index(drop=True)
def smote(X,y):
    mi=np.where(y==1)[0]; ma=np.where(y==0)[0]; n=len(ma)-len(mi)
    i=np.random.choice(mi,n,replace=True); j=np.random.choice(mi,n,replace=True); a=np.random.uniform(.2,.8,n)
    return np.vstack([X,X[i]*a[:,None]+X[j]*(1-a[:,None])]),np.concatenate([y,np.ones(n,int)])
