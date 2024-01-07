# -*- coding = utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
import pyecharts as pye
import seaborn as sns
from scipy.stats import norm

df = ts.get_h_data('000333', start='2017-01-01', end='2018-11-8')
print(df)

'''
start_date='01-07-2015'
end_date='01-07-2017'
united=quandl.get('WIKI/UAL',start_date=start_date,end_date=end_date)
america=quandl.get('WIKI/AAL',start_date=start_date,end_date=end_date)
'''

# 计算日均收益率
df1 = df['close'].sort_index(ascending=True)
df1 = pd.DataFrame(df1)
df1['date'] = df1.index
df1['date'] = df1[['date']].astype(str)
df1["rev"] = df1.close.diff(1)
df1["last_close"] = df1.close.shift(1)
df1["rev_rate"] = df1["rev"] / df1["last_close"]
df1 = df1.dropna()
print(df1.head(10))
# 使用PYECHARTS，画出收盘价曲线
line = pye.Line(title="Mytest")
attr = df1["date"]
v1 = df1["close"]
line.add("000333", attr, v1, mark_point=["average"])
line.render("Line-High-Low.html")

sns.distplot(df1["rev_rate"])

sRate = df1["rev_rate"].sort_values(ascending=True)
p = np.percentile(sRate, (1, 5, 10), interpolation='midpoint')
print(p)

plt.show()


u = df1.rev_rate.mean()
σ2 = df1.rev_rate.var()
σ = df1.rev_rate.std()
Z_01 = norm.ppf(0.01)  # stats.norm.ppf正态分布的累计分布函数的逆函数，即下分位点
# 因为(R* - u)/σ = Z_01
# 所有R* = Z_01*σ + u
print(u + Z_01 * σ)