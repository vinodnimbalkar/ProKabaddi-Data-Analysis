import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

matches = pd.read_csv('MatchResults.csv') 

plt.subplots(figsize=(10,6))
ax = matches['MatchVenue'].value_counts().plot.bar(width=0.5, color='gbbbbbyyrrrr')
plt.title('Match per Venue')
ax.set_xlabel('Grounds')
ax.set_ylabel('Count')
plt.show()