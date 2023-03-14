from jenkspy import JenksNaturalBreaks
import numpy as np
import pandas as pd

def equal_freq_binner(dat: pd.DataFrame, bins: int) -> pd.DataFrame:
    data = pd.DataFrame(columns=dat.columns)
    for i in dat.columns:
        data[i] = pd.qcut(dat[i], q=bins, precision=0, labels=[ "cat"+str(i) for i in range(bins)])
    return data

class jenks_binner:
    def __init__(self, MAX_CLASSES:int=10, THRESHOLD:int=0.8):
        self.MAXCAT = MAX_CLASSES
        self.THRESHOLD = THRESHOLD
        self.jnb = None

    def govf(self, jnb, arr):
        sdam = np.sum((arr - arr.mean()) ** 2)
        sdcm = sum([np.sum((g - g.mean()) ** 2) for g in jnb.groups_])
        return (sdam - sdcm) / sdam

    def binit(self, ser: pd.Series) -> pd.Series:
        opt = self.MAXCAT
        for k in range(2,self.MAXCAT):
            jnb = JenksNaturalBreaks(k)
            arr = np.array(ser)
            jnb.fit(arr)
            thresh = self.govf(jnb,arr)
            if(thresh>self.THRESHOLD):
                opt = k
                break
        self.jnb = JenksNaturalBreaks(opt)
        arr = np.array(ser)
        self.jnb.fit(arr)
        lab = self.jnb.labels_
        breaks = ["-inf"]+ [str(i) for i in self.jnb.inner_breaks_]  + ["+inf"]
        ans = []
        for i in lab:
            ans = ans + [breaks[i] + "< "+ ser.name +" <"+ breaks[i+1]]
        return pd.Series(ans, index=ser.index)

    def fit(self, ser: pd.Series) -> pd.Series:
        var = ser
        var_notna = var[var.notna()]
        var_= self.binit(var_notna)
        var_cleaned = []
        counter = 0
        for i in range(len(var)):
            if(pd.notna(list(var.index)[i])):
                var_cleaned += [var_[list(var_.index)[i-counter]]]
            else:
                var_cleaned += [None]
                counter = counter+1
        return pd.Series(var_cleaned,name=ser.name, index=ser.index)

    def binit_predict(self, ser: pd.Series) -> pd.Series:
        lab = [self.jnb.predict(i) for i in ser]
        breaks = ["-inf"]+ [str(i) for i in self.jnb.inner_breaks_]  + ["+inf"]
        ans = []
        for i in lab:
            ans = ans + [breaks[i] + "< "+ ser.name +" <"+ breaks[i+1]]
        return pd.Series(ans, index=ser.index)

    def transform(self, ser: pd.Series):
        var = ser
        var_notna = var[var.notna()]
        var_ = self.binit_predict(var_notna)
        var_cleaned = []
        counter = 0
        for i in range(len(var)):
            if(pd.notna(list(var.index)[i])):
                var_cleaned += [var_[list(var_.index)[i-counter]]]
            else:
                var_cleaned += [None]
                counter = counter+1
        return pd.Series(var_cleaned,name=ser.name, index=ser.index)