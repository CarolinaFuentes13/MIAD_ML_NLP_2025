#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce

def predict_proba(artists:str, album_name:str, track_name:str, duration_ms:int, explicit:bool,
       danceability:float, energy:float, key:int, loudness:float, mode:int, speechiness:float,
       acousticness:float, instrumentalness:float, liveness:float, valence:float, tempo:float,
       time_signature:int, track_genre:str):

    model = joblib.load(os.path.dirname(__file__) + '/popularity_RF.pkl') 

   
    input_data = {
        "artists": [artists],
        "album_name": [album_name],
        "track_name": [track_name],
        "duration_ms": [duration_ms],
        "explicit": [explicit],
        "danceability": [danceability],
        "energy": [energy],
        "key": [key],
        "loudness": [loudness],
        "mode": [mode],
        "speechiness": [speechiness],
        "acousticness": [acousticness],
        "instrumentalness": [instrumentalness],
        "liveness": [liveness],
        "valence": [valence],
        "tempo": [tempo],
        "time_signature": [time_signature],
        "track_genre": [track_genre]
    }
    
    df = pd.DataFrame(input_data)
    

   
    # Make prediction
    p1 = model.predict(df)

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) != 19:
        print('Verifica que todos los campos se hayan diligenciado')
        
    else:
        artists = sys.argv[1]
        album_name = sys.argv[2]
        track_name = sys.argv[3]
        duration_ms = sys.argv[4]
        explicit = sys.argv[5]
        danceability = sys.argv[6]
        energy = sys.argv[7]
        key = sys.argv[8]
        loudness = sys.argv[9]
        mode = sys.argv[10]
        speechiness = sys.argv[11]
        acousticness = sys.argv[12]
        instrumentalness = sys.argv[13]
        liveness = sys.argv[14]
        valence = sys.argv[15]
        tempo = sys.argv[6]
        time_signature = sys.argv[17]
        track_genre = sys.argv[18]

        p1 = predict_proba(artists, album_name, track_name, duration_ms, explicit,
       danceability, energy, key, loudness, mode, speechiness,
       acousticness, instrumentalness, liveness, valence, tempo,
       time_signature, track_genre)
        
        print('Probability of popularity: ', p1)
        