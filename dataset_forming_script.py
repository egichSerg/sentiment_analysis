#!/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

index = pd.Series()
df = pd.read_csv('/content/sentiment_analysis/sentiment dataset 4/clothing_dataset.csv', on_bad_lines='skip', sep='\t')
df['text'] = df.index
df['index'] = np.arange(0, 90000)
df = df.set_index(['index'])
df = df.rename(columns={'text' : 'text', 'review  sentiment' : 'label'})
df = df[['text', 'label']]

train, test = train_test_split(df, test_size=0.2)

train.to_csv('/content/sentiment_analysis/sentiment dataset 4/train.csv')
test.to_csv('/content/sentiment_analysis/sentiment dataset 4/test.csv')
