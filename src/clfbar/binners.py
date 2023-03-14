def equal_freq_binner(dat: pd.DataFrame, bins: int) -> pd.DataFrame:
    data = pd.DataFrame(columns=dat.columns)
    for i in dat.columns:
        data[i] = pd.qcut(X[i], q=bins, precision=0, labels=["cat"+str(i) for i in range(bins)])

