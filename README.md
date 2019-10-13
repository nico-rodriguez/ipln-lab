# Laboratorio del curso IPLN 2019

Clasificador de humor en mensajes de Twitter.

## Primeros pasos

### Dependencias

Este proyecto usa Python 3 (versión >= 3.5).
El archivo `requirements.txt` contiene una lista de las dependencias del proyecto, que se pueden instalar mediante `pip3`:

```
pip3 install -r requirements.txt
```

Las dependencias principales son:

1. matplotlib==3.0.3
2. numpy==1.17.2
3. pandas==0.25.1
4. scikit-learn==0.21.3
5. tensorflow==2.0.0

### Instancias

Se requiere de tres archivos, llamados `humor_train.csv`, `humor_val.csv` y `humor_test.csv`, necesarios para entrenar y evaluar el clasificador. Estos archivos deben encontrarse en un mismo directorio, que será pasado por parámetro al módulo principal `es_humor.py`.
Opcionalmente, se puede disponer de un conjunto de tests en formato csv.

## Ejecutando el clasificador

El módulo principal es `es_humor.py`.

### Modos de invocación

Invocar como

```
python3 es_humor.py <data_path> test_file1.csv ... test_fileN.csv
```

donde:
1. <data_path> es una ruta al directorio donde se encuentran los archivos `humor_train.csv`, `humor_val.csv` y `humor_test.csv`.
2. `test_file1.csv ... test_fileN.csv` son archivos de test. El clasificador genera un archivo de salida `test_file1.out ... test_fileN.out` para cada uno de ellos.

### Archivos de salida

El formato de los archivos de salida correspondientes a los archivos de test es un conjunto de 0 y 1, separados por saltos de línea. Cada valor corresponde a la clasificación de una instancia.

### Log de desarrollo

#### Parser
* Se agrega el módulo Parser para procesar los csv, obteniendo diccionario de [word, embedding vector] (embedding_index). Se agrega también 2 operaciones las cuales generan para cada word el indice que identifica esa palabra [word, word_index] (word_index) y la matriz de embeddings que asocia un indice de palabra con su vector de embeddings [word_index, embedding vector] (embedding_matrix).
#### Arquitectura
* Se utilizaron redes lstm, gru de la librería Keras junto con los embeddings brindados en el curso. Se intentó luego con arquitecturas híbridas entre las mismas cambiando la cantidad de capas, la cantidad de neuronas por capa y la cantidad de epochs. Algunas de ellas fueron: single_lstm, triple_lstm, bidirectional_lstm, lstm_gru, parallel_bidirectional_lstm_lstm_gru. Se vió que los resultados obtenidos cambiando la arquitectura no afectaba de forma considerable el resultado, pero si los tiempos de ejecución.
 Por lo tanto se decidió por utilizar la más simple posible que obtenga mejores resultados (lstm_gru).
#### Preprocesamiento
* Para el preprocesamiento se intentó removiendo hashtags, retweets, cc, menciones, urls, emojis, además de otros caracteres para los que no había embeddings. Se vió que habían caracteres en otros idiomas, por lo que se utilizó una regex que solo tome caracteres latinos para el preprocesamiento. De igual forma los resultados empeoraron luego de esto, por lo que se volvió al final a no realizar preprocesamiento. Se debió para que las entradas tengan un largo fijo para dar como input a las redes padding si no llegaban los tweets al largo esperado.
##### Stopwords
* Se removieron stop words de los tweets (remove_stop_words), utilizando las stopwords de nltk para español, tanto como stopwords encontradas en internet para español. Luego de utilizarlas se vió que el resultado no era mejor que a no usarlas, por lo que se optó por no utilizarlas.
##### Lemmatización
* Se investigó sobre herramientas para lematizar en español, se vió que si bien nltk proveía lematización para el idioma inglés, no lo hacía para español. Por lo que se descartó dicha opción.
##### Stemming
* Para el stemming se vió que nltk proveía la opción de stemming (SnowBallStemmer). Los resultados obtenidos utilizando dicho método no resultaron mejores a no utilizarlos, por lo que se descartó también.

#### Word Recovery (palabras que no aparecían en el archivo de embeddings)
* Para intentar recuperar las palabras se intentaron varias alternativas, se utilizó el método edit_distance de nltk para encontrar las palabras más similares en el diccionario de palabras de las embeddings, lo cual no generaba muy buenos resultados y aumentaba considerablemente el tiempo de ejecución.

* Se utilizó también como alternativa la operación get_close_matches para obtener palabras similares, teniendo buenos resultados pero teniendo un aumento considerable del tiempo de ejecución si la palabra no estaba en el diccionario.

* Otras alternativas utilizadas involucran pasar a lowercase todas las palabras de los tweets tanto como las palabras de los vectores de embeddings, lo cual no mejoró los resultados por lo que fue descartado.

* Se intentó también generando multiples entradas en los embeddings con distintas versiones de la palabra para aumentar el hit-rate al buscar si la palabra existía, agregando una entrada para la palabra original, una para la palabra en lowercase y otra para la palabra stemmizada. Luego en caso de que no se encuentre ninguna de las 3 versiones de la palabra se utiliza búsqueda por similaridad.

* Las palabras no reconocidas finalmente luego de intentar recuperarlas se las sustituye por <UNK> el cual tiene como vector de embedding el vector nulo. 

#### Optimización de hiper-parametros
* Para la obtención de los hiper-parametros se utilizó GridSearch para mejorar los parametros de: activación, optimizador, init-mode lo cual permitió mejorar las métricas obtenidas.




