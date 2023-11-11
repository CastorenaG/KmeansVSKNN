import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos
df = pd.read_csv("c:\\diabetes_prediction_dataset.csv")

# Convertir variables categóricas a variables dummy
df = pd.get_dummies(df, columns=["gender", "smoking_history"])

# Seleccionar características y etiqueta
X = df.drop("diabetes", axis=1)
y_true = df["diabetes"]

# Dividir el conjunto de datos en entrenamiento y prueba para KNN
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans_clusters = kmeans.fit_predict(X)
conf_matrix_kmeans = confusion_matrix(y_true, kmeans_clusters)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
conf_matrix_knn = confusion_matrix(y_test, knn_predictions)

# Visualizar la matriz de confusión de KMeans con matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(conf_matrix_kmeans, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for KMeans')
plt.colorbar()
classes_kmeans = ['Class 0', 'Class 1']
tick_marks_kmeans = range(len(classes_kmeans))
plt.xticks(tick_marks_kmeans, classes_kmeans, rotation=45)
plt.yticks(tick_marks_kmeans, classes_kmeans)
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Visualizar la matriz de confusión de KNN con matplotlib
plt.subplot(1, 2, 2)
plt.imshow(conf_matrix_knn, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for KNN')
plt.colorbar()
classes_knn = ['Class 0', 'Class 1']
tick_marks_knn = range(len(classes_knn))
plt.xticks(tick_marks_knn, classes_knn, rotation=45)
plt.yticks(tick_marks_knn, classes_knn)
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.tight_layout()
plt.show()

# Comparar resultados
print("Confusion Matrix (KMeans):\n", conf_matrix_kmeans)
print("\nClassification Report (KMeans):\n", classification_report(y_true, kmeans_clusters))

print("\nConfusion Matrix (KNN):\n", conf_matrix_knn)
print("\nClassification Report (KNN):\n", classification_report(y_test, knn_predictions))
