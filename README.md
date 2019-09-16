# Laboratorio del curso IPLN 2019

Clasificador de humor en mensajes de Twitter.

## Primeros pasos

### Dependencias

Este proyecto usa Python 3 (versión >= 3.5).
El archivo `requirements.txt` contiene una lista de las dependencias del proyecto, que se pueden instalar mediante `pip3`:

```
pip3 install -r requirements.txt
```

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
