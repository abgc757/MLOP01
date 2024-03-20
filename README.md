# <h1 align=center> **Sistema de Recomendación**</h1>
# <h2> **Proyecto individual Nº1**</h1>
# `Machine Learning Operations (MLOps)`
## **Henry's Labs**
### *Por Abraham Gómez DataPT01*

El objetivo principal del proyecto es crear un sistema de recomendación de videojuegos utilizando técnicas de aprendizaje automático (Machine Learning). El proyecto se divide en diferentes etapas:

1. Extracción, transformación y carga (ETL): Se realiza la extracción de datos del conjunto de datos "dataset\australian_user_reviews.json", "dataset\australian_users_items.json" y "dataset/output_steam_games.json". Se aplican transformaciones para estandarizar la "data" debido a que los registros JSON contienen errores, es por ello que deben de tranformarse y con el objetivo de economizar espacio de almacenamiento se procede a archivar la data en formato parquet. En el proceso de transformación se desanidan los items, recomendaciones y se procesan las recomendaciones (reviews) en una variable discreta "Sentiment_analisys", ademas es necesario eliminar algunos registros nulos que no aportan información relevante.

2. Desarrollo de API: El desarollo de la API inicia con la creación de funciones las cuales fueron probadas en su fase inicial en el notebook `funciones.ipynb` las cuales fueron integradas posteriormente en `main.py` para su ejecución en la por medio de FastAPI. Las funcionalidades disponibles son:
- PlayTimeGenre(genero : str)
- UserForGenre(genero : str)
- UsersRecommend(año : int)
- UsersNotRecommend( año : int)
- sentiment_analysis( año : int)
- recomendacion_juego(id : int):

3. Análisis exploratorio de los datos (EDA): Se realiza un análisis exploratorio del [Dataset](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj). Se explora cada uno de los datasets y se normalizan los datos encontrado las siguientes particularidades para los diferentes archivos

4. Sistema de recomendación: Se implementa un sistema de recomendación dado el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado

## Tabla de contenidos

- [ETL (Extracción, Transformación y Carga)](#etl-extracción-transformación-y-carga)
- [Desarrollo de API](#desarrollo-de-api)
- [Análisis Exploratorio de los Datos (EDA)](#análisis-exploratorio-de-los-datos-eda)
- [Sistema de Recomendación](#sistema-de-recomendación)

## ETL (Extracción, Transformación y Carga)

1. Lectura y Procesamiento de Datos:

- Lee los datos de varios archivos JSON y los convierte en DataFrames de Pandas. Estos DataFrames contienen información sobre reseñas de usuarios, juegos de Steam y géneros de juegos.
- Realiza algunas limpiezas de datos, como eliminar registros duplicados y reemplazar valores nulos.
- Normalizar los datos como fechas, valores enteros o decimales
2. Análisis de Sentimiento:

- Utiliza la biblioteca `NLTK` para realizar un análisis de sentimiento de las reseñas de los usuarios. Asigna un valor de sentimiento (positivo, neutral o negativo) a cada reseña.

3. Generacion de tablas segun la funcionalidad requerida para la API:

- `ranking_genre.parquet` contiene el total de horas jugadas por genero y año
- `user_genre.parquet` este dataset contiene las horas jugadas por usuario ,genero y año.
- `recommend.parquet` el dataset contiene una lista de juegos los cuales fueron recomendado y copntiene recomendaciones positivas o neutrales (`reviews.recommend` = `True` y comentarios negativos) 
- `notrecommend.parquet` contiene aquellos juegos que no fueron recomendados por los usuarios y/o contienen recomendaciones negativas (`reviews.recommend` = `False` y comentarios negativos)
- `sentiments.parquet` contiene el total de reseñas positivas (`2`), neutras (`1`) y negativas (`0`) de la columna `sentiment_analysis` por año.
- `reviews_per_item.parquet` Este dataset se obtiene realizando las siguientes acciones: **Agrupación de reseñas por ítem**: El código recorre todas las reseñas en `n_reviews` y agrupa las reseñas por `item_id`, es decir, por videojuego. Cada `item_id` se mapea a una lista de reseñas correspondientes. **Concatenación de reseñas**: Para cada videojuego, se concatenan todas sus reseñas en una sola cadena de texto. **Vectorización TF-IDF**: Se aplica la vectorización TF-IDF a las reseñas concatenadas. La vectorización TF-IDF convierte el texto en un vector numérico que puede ser utilizado para el análisis de texto y machine learning. Se seleccionan las 1000 palabras más importantes para formar el vector. **Creación de DataFrame de vectores de reseñas**: Se crea un nuevo DataFrame que contiene el `id` del videojuego y su vector de reseñas correspondiente. Este DataFrame se guarda en un archivo parquet para su uso posterior. **Preparación de datos para recomendaciones de videojuegos**: Se seleccionan las columnas `item_id` y `sentiment_analysis` de `n_reviews` para crear un nuevo DataFrame `games_recommend`. Se eliminan las filas con valores nulos y se llenan los valores nulos restantes con 0. Este DataFrame también se guarda en un archivo parquet para su uso posterior.


## Desarrollo de API

En esta sección se detalla el desarrollo de la API que permite interactuar con el sistema de recomendación y acceder a diferentes funciones y servicios.

### Funciones disponibles

Las siguientes funciones están disponibles a través de la API:

1. **def PlayTimeGenre(genero : str)**
    Esta función devuelve el año de lanzamiento con más horas jugadas para un género de videojuego específico.

    Parámetros:
    genero (str): El género del videojuego. Debe ser una cadena de texto que represente un género de videojuego válido.

    Devuelve:
    dict: Un diccionario que contiene el año de lanzamiento con más horas jugadas para el género dado. 
          La clave del diccionario es una cadena de texto que dice "Año de lanzamiento con más horas jugadas para Género {genero}", 
          y el valor es un entero que representa el año. 
          Si el género dado no existe, el diccionario devuelto tendrá la clave "Error" y el valor "El género no existe".

    Ejemplo:
    >>> PlayTimeGenre('Action')
    {'Año de lanzamiento con más horas jugadas para Género Action': 2005}
    

2. **UserForGenre(genero : str)**
    Esta función devuelve el usuario que ha jugado más horas en un género de videojuego específico.

    Parámetros:
    genero (str): El género del videojuego. Debe ser una cadena de texto que represente un género de videojuego válido.

    Devuelve:
    dict: Un diccionario que contiene el usuario que ha jugado más horas para el género dado. 
          La clave del diccionario es una cadena de texto que dice "Usuario con más horas jugadas para Género {genero}", 
          y el valor es una lista de diccionarios, donde cada diccionario representa un año y las horas jugadas en ese año por el usuario. 
          Si el género dado no existe, el diccionario devuelto tendrá la clave "Error" y el valor "El género no existe".

    Ejemplo:
    >>> UserForGenre('Action')
    {'Usuario con más horas jugadas para Género Action': 'user123', [{'Año:2005, Horas:120'}, {'Año:2006, Horas:150'}]}
    """
3. **UsersRecommend(año : int)**
    Esta función devuelve los tres juegos más recomendados para un año específico.

    Parámetros:
    año (int): El año para el cual se buscan las recomendaciones. Debe ser un entero que represente un año válido.

    Devuelve:
    list: Una lista de diccionarios que contiene los tres juegos más recomendados para el año dado. 
          Cada diccionario tiene la clave "Puesto {i+1}" y el valor es el título del juego. 
          Si no hay juegos recomendados para el año dado, se devuelve un diccionario con la clave "No hay juegos recomendados para el año {año}".

    Ejemplo:
    >>> UsersRecommend(2020)
    [{'Puesto 1' : 'Juego A'}, {'Puesto 2' : 'Juego B'}, {'Puesto 3' : 'Juego C'}]
def UsersNotRecommend(año : int):
    """
    Esta función devuelve los tres juegos con las peores recomendaciones para un año específico.

    Parámetros:
    año (int): El año para el cual se buscan las recomendaciones. Debe ser un entero que represente un año válido.

    Devuelve:
    list: Una lista de diccionarios que contiene los tres juegos con las peores recomendaciones para el año dado. 
          Cada diccionario tiene la clave "Puesto {i+1}" y el valor es el título del juego. 
          Si no hay juegos con malas recomendaciones para el año dado, se devuelve un diccionario con la clave "No hay registros de juegos con peores recomendaciones para el año {año}".

    Ejemplo:
    >>> UsersNotRecommend(2020)
    [{'Puesto 1' : 'Juego A'}, {'Puesto 2' : 'Juego B'}, {'Puesto 3' : 'Juego C'}]
4. **UsersNotRecommend(año : int)**
    Esta función devuelve los tres juegos con las peores recomendaciones para un año específico.

    Parámetros:
    año (int): El año para el cual se buscan las recomendaciones. Debe ser un entero que represente un año válido.

    Devuelve:
    list: Una lista de diccionarios que contiene los tres juegos con las peores recomendaciones para el año dado. 
          Cada diccionario tiene la clave "Puesto {i+1}" y el valor es el título del juego. 
          Si no hay juegos con malas recomendaciones para el año dado, se devuelve un diccionario con la clave "No hay registros de juegos con peores recomendaciones para el año {año}".

    Ejemplo:
    >>> UsersNotRecommend(2020)
    [{'Puesto 1' : 'Juego A'}, {'Puesto 2' : 'Juego B'}, {'Puesto 3' : 'Juego C'}]

5. **sentiment_analysis(año : int)**
    Esta función realiza un análisis de sentimientos de las reseñas de videojuegos para un año específico.

    Parámetros:
    año (int): El año para el cual se realiza el análisis de sentimientos. Debe ser un entero que represente un año válido.

    Devuelve:
    dict: Un diccionario que contiene el recuento de reseñas negativas, neutrales y positivas para el año dado. 
          La clave del diccionario es una cadena de texto que dice "Negative={n}, Neutral={n}, Positive={n}", 
          donde {n} es el número de reseñas de cada tipo. 

    Ejemplo:
    >>> sentiment_analysis(2020)
    {'Negative=500, Neutral=200, Positive=300'}

6. **recomendacion_juego(id)**
    Esta función devuelve una lista de juegos recomendados basados en la similitud de las reseñas con un juego específico.

    Parámetros:
    id (int): El ID del juego para el cual se buscan recomendaciones. Debe ser un entero que represente un ID de juego válido.

    Devuelve:
    dict: Un diccionario que contiene una lista de juegos recomendados para el juego dado. 
          La clave del diccionario es una cadena de texto que dice "Los juegos recomendados para {id} son:", 
          y el valor es una lista de IDs de los juegos recomendados. 

    Ejemplo:
    >>> recomendacion_juego(123)
    {'Los juegos recomendados para 123 son:': [456, 789, 1011, 1213, 1415]}



### Sistema de recomendación.


El sistema de recomendación funciona en dos partes principales: la función `similarityCosine` y la función `recomendacion_juego`.

1. **Función `similarityCosine`**: Esta función toma dos vectores como entrada y calcula la similitud del coseno entre ellos. La similitud del coseno es una medida de la similitud entre dos vectores en un espacio multidimensional. Se calcula como el producto escalar de los vectores dividido por el producto de sus normas (o longitudes). Si alguna de las normas es cero, la función devuelve cero. La ecuación aplicada en esta función es:

$$
\text{similitud}(a, b) = \frac{a \cdot b}{||a||_2 \cdot ||b||_2}
$$

Donde:
- `a · b` es el producto escalar de `a` y `b`.
- `||a||_2` y `||b||_2` son las normas euclidianas de `a` y `b`.

2. **Función `recomendacion_juego`**: Esta función toma un ID de juego como entrada y devuelve una lista de juegos recomendados basados en la similitud del coseno de las reseñas de los juegos.

    - Primero, la función carga un conjunto de datos de reseñas y selecciona las reseñas para el juego dado.
    - Luego, calcula la similitud del coseno entre la primera reseña del juego dado y todas las demás reseñas en el conjunto de datos.
    - Después de calcular las similitudes, la función ordena los juegos por similitud del coseno en orden descendente y selecciona los 10 primeros.
    - Finalmente, la función devuelve una lista de los IDs de los juegos recomendados, excluyendo el juego original.

### Integración en FastAPI

El desarrollo de la API se ha realizado utilizando el framework FastAPI. El archivo `main.py` contiene la implementación de las funciones descritas anteriormente. Estas funciones estan en el notebook `funciones.ipynb` donde se realizaron los primeros desarrollos y pruebas.

Para integrar estas funciones en FastAPI y hacer que estén disponibles a través de la API, se ha implementado el archivo `main.py`. Este archivo define las rutas y los endpoints correspondientes a cada función. Además, se han establecido las validaciones necesarias para los parámetros de entrada y se han definido las respuestas esperadas.
.


## Análisis Exploratorio de los Datos (EDA)

### `australian_user_reviews.json`
La data se encuentra anidada en formato JSON para la mayoria de los archivos proporcionados en el archivo despues de desanidar los datos y nos encotramos con las siguientes columnas:
- `funny`: Expresa la cantidad de votos que indican que la review es divertida
- `posted`: Fecha de publicación de la recomendación
- `last_edited`: Última fecha de edición de la recomendación
- `item_id`: Identificador unico del videojuego
- `helpful`: Indica si el usuario ha encontrado útil la review se procesa la columna y se procede a generar 3 columnas que continenen el número de personas que valoraron como positivo el comentario (`found_helpful`) del total de usuarios que consultaron la review (`total_people`) y por último la proporcion de usuarios que encontraron util la recomendacion (`percentage`)
-  `recommend`: Indica si el usuario recomienda (True/False) al juego
- `review`: recomendación del usuario, para facilitar  el análisis de los comentarios se ha creado una nueva columna llamada `sentiment_analysis` la es discretizada con 3 posibles valores: 0; si el comentario es negativo, 1; si el comentario es neutro y 2; si el comentario es positivo.
- `user_id`: Identificador único del usuario que realizó la review

La gráfica de dispersión `found_helpful `muestra la relación entre el índice de reseñas y la cantidad de usuarios que encontraron útil cada reseña de videojuegos. La mayoría de los puntos se concentran cerca del eje horizontal, lo que indica que un número menor de usuarios encontró útiles esas reseñas. Sin embargo, hay algunos puntos dispersos hacia arriba, lo que sugiere que algunas reseñas fueron consideradas útiles por un mayor número de usuarios. Esto puede indicar que ciertas reseñas son de alta calidad y han sido valiosas para muchos usuarios.

![Dispersion de voto de reseñas](/images/dispersion_foundhelpful.png)

El histograma muestra la distribución de las fechas de publicación desde el año 2010 hasta el 2015. Se observa una baja frecuencia de publicación en los años 2010 a 2013, con un incremento notable en 2014 y un pico significativo en 2015. Esto sugiere que hubo un aumento considerable en la cantidad de datos o elementos publicados hacia el final del período representado.

![Distribucion de recomendaciones](/images/histograma_posted.png)

La gráfica de barras que representa el tipo de recomendaciones publicadas por los usuarios, clasificadas en "Positivo", "Neutro" y "Negativo". La barra "Positivo" es significativamente más alta, indicando una mayor cantidad de recomendaciones positivas en comparación con las neutras y negativas. Esto sugiere que la mayoría de los usuarios han tenido experiencias favorables y han compartido reseñas positivas.

![Tipo de recomendaciones](/images/tipo_recomendacion.png)

### `australian_users_items.json`
El dataset contiene información relevante a los videojuegos que cada usuario tiene asociado a su cuenta de STEAM, en la columna `items` hay una lista que contiene la siguiente información:
- `item_id`: Identificador único del videojuefo en la base de datos de STEAM
- `item_name`: Nombre del videojuego
- `playtime_forever`: Representa el tiempo total de juego acumulado “registrado” en minutos
- `playtime_2weeks`: Indica la cantidad de minutos que un usuario ha jugado en las últimas dos semanas

El histograma muestra la distribución del tiempo total de juego de los usuarios. Las barras representan la frecuencia de usuarios según los minutos totales jugadas, en una escala logarítmica. La mayoría de los jugadores han acumulado alrededor de 10^3 (mil) minutos.

![Distribución de tiempo de juego](/images/histograma_playtimeforever.png)

El gráfico de dispersión representa la relación entre el índice de cada usuario y su tiempo total de juego en horas. La concentración densa de puntos cerca del eje horizontal sugiere que la mayoría de los usuarios tienen un tiempo total de juego más bajo, mientras que los puntos dispersos hacia arriba indican que hay usuarios con tiempos de juego significativamente más altos. Este patrón puede reflejar que hay jugadores casuales que juegan menos horas y jugadores más dedicados o "hardcore" que invierten mucho más tiempo en los juegos de la plataforma Steam.

![Distribución de tiempo de juego](/images/dispersion_playtimeforever.png)

### `output_steam_games.json`

Por último este dataset contiene la información de los juegos que estan disponibles en STEAM, en este archivo podemos notar que existe una gran cantidad de registros nulos, se proceden a borrar los registros nulos siempre y cuando este nula la columna `title` y `app_name`

El dataset contiene las siguientes columnas:
- `publisher`: Empresa que publica el videojuego
- `genres`: Lista de generos que describe el videojuego
- `app_name`: Nombre de la aplicación
- `title`: Título del videojuego
- `url`: Dirección url del videojuego
- `release_date`: Fecha en que se lanzó el juego.
- `tags`: Lista de etiquetas que describen y categorizan al videojuego
- `reviews_url`: Url de las recomendaciones realizadas por los usuarios
- `specs`: Especificaciones del videojuego
- `price`: Precio de venta
- `early_access`: Indica si el videojuego está disponible bajo el modelo de “acceso anticipado”
- `id`: Idenitificador del  videojuego, único para cada uno, coincide con `item_id` para los otros datasets
- `developer`: Desarrollador

El gráfico de dispersión de precios muestra la variabilidad en los costos de los videojuegos, con una concentración densa de puntos cerca del eje horizontal, indicando que muchos juegos tienen precios bajos. Algunos puntos se extienden hacia valores más altos, lo que sugiere que hay juegos con precios más elevados.

![Dispersión de precios en videojuegos](/images/dispersion_price.png)


### Deploy
Para realizar un deploy del proyecto se utilizó [render.com/](https://render.com/) como plataforma. Puede probar el deploy en 
https://mlop01.onrender.com/docs

### Video del proyecto
Consulte el video del proyecto desde [Youtube](https://youtu.be/-XHZDQSwNH4)

https://youtu.be/-XHZDQSwNH4