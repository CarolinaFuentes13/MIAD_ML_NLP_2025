#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

# ---------- BERT Vectorizer ----------
class BERTVectorizer:
    def __init__(self, model_name='distilbert-base-uncased', max_length=256):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.max_length = max_length
        self.model.eval()

    def transform(self, texts):
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts, desc="Generando embeddings BERT"):
                inputs = self.tokenizer(text, return_tensors="pt", padding="max_length",
                                        truncation=True, max_length=self.max_length)
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(cls_embedding)
        return np.array(embeddings)



def predict_proba(title:str, plot:str):

    model = joblib.load(os.path.dirname(__file__) + '/genero_pelicula.pkl') 
    
    input_data = {
        "title": [title],
        "plot": [plot],
    }
    
    df = pd.DataFrame(input_data)
    df['text'] = df['title'] + ' ' + df['plot']

    vectorizer = BERTVectorizer()
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
        