## Análisis de Predicción de Diabetes

### Descripción

Este repositorio contiene un script en Python para analizar un conjunto de datos de predicción de diabetes. El script utiliza bibliotecas populares de ciencia de datos como Pandas, Matplotlib y scikit-learn para realizar clustering (KMeans) y clasificación (KNN) en el conjunto de datos. El objetivo es evaluar y comparar el rendimiento de ambos modelos en la predicción de la diabetes.

### Instrucciones

1. **Clonar el Repositorio:**
   
2. **Instalar Dependencias:**
   ```bash
   pip install pandas matplotlib scikit-learn
   ```

3. **Ejecutar el Script:**
   ```bash
  algoritmos.py

### Archivos del Proyecto

- **`algoritmospy.py`**: El script principal en Python que contiene el análisis.
- **`datos_prediccion_diabetes.csv`**: El conjunto de datos utilizado para el análisis.

### Proceso de Análisis

1. **Cargar y Preprocesar Datos:**
   - Cargar el conjunto de datos de diabetes desde un archivo CSV.
   - Convertir variables categóricas en variables dummy.

2. **Dividir los Datos:**
   - Dividir el conjunto de datos en características (`X`) y etiquetas (`y_true`).
   - Crear conjuntos de entrenamiento y prueba para el modelo KNN.

3. **KMeans (Clustering):**
   - Aplicar el algoritmo de clustering KMeans para agrupar los datos en dos clusters.
   - Calcular y visualizar la matriz de confusión para KMeans.

4. **KNN (Clasificación):**
   - Utilizar el algoritmo de clasificación KNN para predecir la diabetes.
   - Calcular y visualizar la matriz de confusión para KNN.

5. **Visualizar Resultados:**
   - Utilizar Matplotlib para representar visualmente las matrices de confusión de KMeans y KNN.

6. **Comparar Resultados:**
   - Imprimir matrices de confusión e informes de clasificación para comparar el rendimiento de KMeans y KNN.

### Requisitos

- Python 3.x
- Pandas
- Matplotlib
- scikit-learn

Siéntete libre de explorar y modificar el script para tus propios conjuntos de datos o análisis. ¡Si encuentras útil esto, no olvides darle una estrella!
