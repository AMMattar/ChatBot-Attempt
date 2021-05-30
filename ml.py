import pickle
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
from liberary import *


df = pd.read_csv('data.txt', sep='\t')
a = pd.Series(df.columns)
a = a.rename({0: df.columns[0], 1: df.columns[1]})

df = df.append(a, ignore_index=True)
df.columns = ['Questions', 'Answers']


pipe.fit(df['Questions'], df['Answers'])
# wh = Pipe.predict(['What are you doing'])[0]
# print(wh)
pickle.dump(pipe, open('Chat.pkl', 'wb'))
