Análisis de Predicción de Diabetes
Este repositorio contiene un script en Python para analizar un conjunto de datos de predicción de diabetes. El script utiliza bibliotecas populares de ciencia de datos como Pandas, Matplotlib y scikit-learn para realizar clustering (KMeans) y clasificación (KNN) en el conjunto de datos. El objetivo es evaluar y comparar el rendimiento de ambos modelos en la predicción de la diabetes.

Instrucciones:
Clonar el Repositorio:

bash
Copy code
git clone https://github.com/tu-nombre/analisis-prediccion-diabetes.git
cd analisis-prediccion-diabetes
Instalar Dependencias:

bash
Copy code
pip install pandas matplotlib scikit-learn
Ejecutar el Script:

bash
Copy code
python analisis_prediccion_diabetes.py
Descripción del Código:
analisis_prediccion_diabetes.py: El script principal en Python que contiene el análisis.
conjunto_datos_prediccion_diabetes.csv: El conjunto de datos utilizado para el análisis.
Descripción:
Cargar y Preprocesar Datos:

Cargar el conjunto de datos de diabetes desde un archivo CSV.
Convertir variables categóricas en variables dummy.
Dividir los Datos:

Dividir el conjunto de datos en características (X) y etiquetas (y_true).
Crear conjuntos de entrenamiento y prueba para el modelo KNN.
KMeans (Clustering):

Aplicar el algoritmo de clustering KMeans para agrupar los datos en dos clusters.
Calcular y visualizar la matriz de confusión para KMeans.
KNN (Clasificación):

Utilizar el algoritmo de clasificación KNN para predecir la diabetes.
Calcular y visualizar la matriz de confusión para KNN.
Visualizar Resultados:

Utilizar Matplotlib para representar visualmente las matrices de confusión de KMeans y KNN.
Comparar Resultados:

Imprimir matrices de confusión e informes de clasificación para comparar el rendimiento de KMeans y KNN.
Requisitos:
Python 3.x
Pandas
Matplotlib
scikit-learn
Siéntete libre de explorar y modificar el script para tus propios conjuntos de datos o análisis. ¡Si encuentras útil esto, no olvides darle una estrella!
