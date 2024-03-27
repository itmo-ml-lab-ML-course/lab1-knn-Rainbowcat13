import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def train_test_val_data():
    series = pd.read_csv('series.csv', index_col=[0])

    le = LabelEncoder()
    ohe = OneHotEncoder(handle_unknown='ignore')

    series = series.dropna()
    series = series[(series['rating'] > 0) &
                    (series['kinopoisk_rating'] > 0) &
                    (series['imdb_rating'] > 0)]
    series['total_rating'] = (series['rating'] * 2 + series['kinopoisk_rating'] + series['imdb_rating']) / 3

    series['total_rating'] = [round(x) for x in series['total_rating']]
    series.drop(columns=['rating', 'kinopoisk_rating', 'imdb_rating'], inplace=True)

    for i in range(3):
        series[f'genre_{i + 1}'] = [genre.split(',  ')[i].lower()
                                    if isinstance(genre, str) and len(genre.split(',  ')) > i
                                    else 'NS' for genre in series['genre']]
    series.drop(columns=['genre'], inplace=True)

    series['start_date'] = pd.to_datetime(series['start_date'], format='%Y-%m-%d')
    series['finish_date'] = pd.to_datetime(series['finish_date'], format='%Y-%m-%d')
    series['days_production'] = [(finish - start).days
                                 for start, finish in zip(series['start_date'], series['finish_date'])]
    series.drop(columns=['start_date', 'finish_date'], inplace=True)

    le.fit(pd.concat([series['genre_1'], series['genre_2'], series['genre_3']]))
    for i in range(1, 4):
        series[f'genre_{i}'] = le.transform(series[f'genre_{i}'])
    series['country'] = le.fit_transform(series['country'])

    countries = pd.DataFrame(ohe.fit_transform(series['country'].values.reshape(-1, 1)).toarray())
    countries = countries.rename(lambda x: f'country_{x}', axis='columns')
    genres = pd.DataFrame(ohe.fit_transform(pd.concat([series['genre_1'],
                                                       series['genre_2'],
                                                       series['genre_3']]).values.reshape(-1, 1)).toarray())
    genres = genres.rename(lambda x: f'genre_{x}', axis='columns')

    series.drop(columns=['country', 'tv_channel', 'genre_1', 'genre_2',
                         'genre_3', 'title'], inplace=True)

    scaler = StandardScaler(with_std=False)
    series = series.join(countries).join(genres)

    y = series['total_rating']
    X = scaler.fit_transform(series.drop(columns=['total_rating']))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        # random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        # random_state=42
    )

    return X_train, y_train, X_val, y_val, X_test, y_test