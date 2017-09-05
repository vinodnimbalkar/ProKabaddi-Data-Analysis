import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

plb = pd.read_csv('PlayerLeaderBoard.csv')

plt.subplots(figsize=(11,6))
max_raids=plb.groupby(['Name'])['Value'].sum()//2
ax=max_raids.sort_values(ascending=False)[:10].plot.bar(width=0.8,color='rgby')
plt.title('Players Total Point in All Season')
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+1),fontsize=11)
plt.show()