import tushare as ts
import pandas as pd

mt = ts.get_hist_data('000001',start='2018-01-01')
df = pd.DataFrame(mt)
print(df)
#df.to_csv('maotai.csv')
