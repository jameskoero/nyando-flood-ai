import pandas as pd
def build(df):
    df=df.copy()
    df['rain_cat']=pd.cut(df['rainfall_3day'],bins=[0,90,110,130,250],labels=[0,1,2,3]).astype(float)
    df['flood_plain']=df['elevation']/(df['slope'].clip(.1)+1)
    df['soil_perm']=1/(df['clay_percent'].clip(1)+1)
    return df
