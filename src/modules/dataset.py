import pandas as pd
import numpy as np

def get_values_dataset():
    df = pd.read_csv('data.csv', index_col=0)

    # Crear variables
    X = df.drop(labels=['song_title', 'artist', 'target'], axis=1).values.tolist()
    y = df['target'].values.tolist()

    return (X,y)

