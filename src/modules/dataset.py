import pandas as pd
import numpy as np

def get_values_dataset():
    df = pd.read_csv('data.csv', index_col=0)

    # Crear variables
    X = df.drop(labels=['song_title', 'artist', 'target'], axis=1)
    y = df['target']

    cols_to_normalize = ['duration_ms', 'key', 'tempo', 'time_signature']

    # Normalizar el conjunto de entrada
    for col in cols_to_normalize:
      X[col] = df[col] / df[col].max()

    X.loudness = X.loudness / X.loudness.min()

    return (X.values.tolist(), y.values.tolist())
