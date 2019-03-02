import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('train.csv')

y = train.SalePrice
x = train.GarageArea

plt.scatter(x,y,alpha=1)
plt.show()



data = pd.concat([train['SalePrice'], train['GarageArea']], axis=1)
data.plot.scatter(x='GarageArea', y='SalePrice', ylim=(0,800000));

z = np.abs(stats.zscore(data))

data1 = data[(z < 3).all(axis=1)]
print(data)
print(data1)
x1=data1.GarageArea
y1=data1.SalePrice
plt.scatter(x1,y1,alpha=1)
plt.show()
