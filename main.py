import pandas as pd
import json
import numpy as np
from fastapi import FastAPI

app = FastAPI()

#Función Similitud de coseno
def similarityCosine(vector1, vector2):
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0 
    else:
        return np.dot(vector1, vector2) / (norm1 * norm2)

@app.get("/")
def index():
    return {'Mensaje': 'Sistema de recomendacion'}

@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre( genero : str ):
    data = pd.read_parquet('dataset/genres_playtime.parquet')
    result = data[data['genres'] == genero].reset_index().head(1)
    del data
    return {f"Año de lanzamiento con más horas jugadas para {genero}" : result['year'].loc[0].item()}

@app.get("/UserForGenre/{genero}")
def UserForGenre( genero : str ):
    data = pd.read_parquet('dataset/user_playtime2genres.parquet')
    data = data[data['genres'] ==genero]
    usuario = data.groupby(['user_id'])['playtime_forever'].sum().sort_values(ascending=False).reset_index()
    result = data[data['user_id'] == usuario['user_id'].loc[0]].reset_index()
    i = 0
    res = {f"Usuario con más horas jugadas para {genero}": usuario['user_id'].loc[0],"Horas jugadas":[]}
    while i<= len(result['user_id']) -1:
        res['Horas jugadas'].append({"Año":result['year'].loc[i].item(), "Horas": result['playtime_forever'].loc[i].item()})
        i = i + 1
    del data
    del usuario
    del result
    return [res]

@app.get("/UsersRecommend/{anio}")
def UsersRecommend( año : int ):
    data = pd.read_parquet('dataset/user_recommend.parquet')
    # Filtrar los datos para el año solicitado
    try:    
        filter = data[(data['year'] == año) & (data['recommend'] == True) & (data['sentiment_analysis'] > 0)]
        list = filter.groupby(by=['item_id']).size().sort_values(ascending=False)
        ids = list.index
        A = filter[filter['item_id'] == ids[0]]['title'].reset_index().loc[0]
        B = filter[filter['item_id'] == ids[1]]['title'].reset_index().loc[0]
        C = filter[filter['item_id'] == ids[2]]['title'].reset_index().loc[0]
        result = [{"Puesto 1" : A['title']},{"Puesto 2" : B['title']}, {"Puesto 3" : C['title']}]
        return result
    except IndexError:
        return [{f"No hay registros para el año {año}"}]

@app.get("/UsersNotRecommend/{anio}")
def UsersNotRecommend( año : int ):
    data = pd.read_parquet('dataset/user_recommend.parquet')
    # Filtrar los datos para el año solicitado
    try:    
        filter = data[(data['year'] == año) & (data['recommend'] == False) & (data['sentiment_analysis'] == 0)]
        list = filter.groupby(by=['item_id']).size().sort_values(ascending=False)
        ids = list.index
        A = filter[filter['item_id'] == ids[0]]['title'].reset_index().loc[0]
        B = filter[filter['item_id'] == ids[1]]['title'].reset_index().loc[0]
        C = filter[filter['item_id'] == ids[2]]['title'].reset_index().loc[0]
        result = [{"Puesto 1" : A['title']},{"Puesto 2" : B['title']}, {"Puesto 3" : C['title']}]
        return result
    except IndexError:
        return [{f"No hay registros para el año {año}"}]
    
@app.get("/Sentiment_Analysis/{anio}")
def Sentiment_Analysis( año : int ):
    data = pd.read_parquet('dataset/user_recommend.parquet')
    try:
        filter = data[data['year'] == año]
        group = filter.groupby(by=['sentiment_analysis']).size()
        return {"Negative =": group[0].item(), "Neutral =": group[1].item(), "Positive =": group[2].item()}
    except:
        return {"Error" : "No hay datos para el año ingresado"}

@app.get("/recomendacion_juego/{id}")
def recomendacion_juego(id : int):
    reviews  = pd.read_parquet('dataset/reviews_per_item.parquet')
    filtro = reviews[reviews['id'] == id]
    try:
        fila1 = filtro.iloc[0]
    except IndexError:
        return {"error": "No se encontraron datos para el ID del juego dado."}   
    fila1 = filtro.iloc[0]
    lista = []
    i = 0
    while i < len(reviews):
        fila2 = reviews.iloc[i]
        lista.append(similarityCosine(fila1,fila2))
        i += 1
    reviews['similarity_cosine'] = lista
    recomendacion =reviews[['id','similarity_cosine']].sort_values(by=['similarity_cosine'], ascending=False).head(10)
    del reviews
    rec = []
    j = 0
    while j < 6:
        if recomendacion['id'].iloc[j] != id:
            rec.append(recomendacion['id'].iloc[j])
        j+=1
    return {f"Los juegos recomendados para {id} son: {rec}"}