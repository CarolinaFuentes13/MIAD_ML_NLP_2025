#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer

class CountVectorizerEmbedder:
    def __init__(self, max_features=1000):
        self.vectorizer = CountVectorizer(max_features=max_features)

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()



def predict_proba(title:str, plot:str):

    model = joblib.load(os.path.dirname(__file__) + '/genero_pelicula.pkl') 
    
    input_data = {
        "title": [title],
        "plot": [plot],
    }
    
    df = pd.DataFrame(input_data)
    df['text'] = df['title'] + ' ' + df['plot']

    vectorizer = CountVectorizerEmbedder()
    text_vec = vectorizer.transform(df['text'])
    
    # Make prediction
    p1 = model.predict(text_vec)

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print('Verifica que todos los campos se hayan diligenciado')
        
    else:
        title = sys.argv[1]
        plot = sys.argv[2]

        p1 = predict_proba(title, plot)

        # Paso 1: Convertir probabilidades a 0s y 1s (puedes ajustar el umbral si es necesario)
        y_pred_bin = (p1 >= 0.4).astype(int)


        # Paso 2: Obtener los géneros con el binarizador original
        # Carga el binarizador previamente guardado
        mlb = joblib.load('mlb_encoder.pkl')

        # Supongamos que y_pred_bin es la predicción binarizada del modelo
        predicted_genres = mlb.inverse_transform(y_pred_bin)

        
        #print('El género de la pelicula es: ', predicted_genres)
        print('El género de la pelicula es: ', predicted_genres)
        