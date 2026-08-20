import os
import sys
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, variance
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import numpy as np
from operator import add

# CONFIGURACIÓN DE SPARK

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder \
    .appName("KMeansAnomalyDetection") \
    .master("local[*]") \
    .config("spark.driver.memory", "10g") \
    .getOrCreate()

sc = spark.sparkContext
print("Spark Context iniciado")

# CARGA Y PREPROCESAMIENTO DE DATOS

input_path_of_file = "kddcup.data"
data_raw = sc.textFile(input_path_of_file, 12)


def parseVector(line):
    columns = line.split(',')
    thelabel = columns[-1]
    featurevector = columns[:-1]
    featurevector = [element for i, element in enumerate(featurevector) if i not in [1, 2, 3]]
    featurevector = np.array(featurevector, dtype=np.float64)
    return (thelabel, featurevector)


labelsAndData_rdd = data_raw.map(parseVector).cache()

sample = labelsAndData_rdd.first()
num_features = len(sample[1])

feature_cols = [f"feature_{i}" for i in range(num_features)]

labelsAndData_rdd_lists = labelsAndData_rdd.map(lambda x: (x[0], x[1].tolist()))

labelsAndData_df = labelsAndData_rdd_lists.map(
    lambda x: Row(label=x[0], features_list=x[1])
).toDF()

for i in range(num_features):
    labelsAndData_df = labelsAndData_df.withColumn(
        f"feature_{i}", 
        col("features_list")[i]
    )

labelsAndData_df = labelsAndData_df.drop("features_list")
labelsAndData_df.cache()

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(labelsAndData_df)

thedata = df_assembled.select("features").cache()
n = thedata.count()

# ENTRENAMIENTO INICIAL DE K-MEANS CON 2 CLUSTERS

kmeans = KMeans(k=2, maxIter=10, initMode="random", seed=42)
k_clusters = kmeans.fit(thedata)


def getFeatVecs(df_with_features, feature_column_names):
    agg_expressions = [variance(c).alias(c) for c in feature_column_names]
    variance_row = df_with_features.agg(*agg_expressions).first()
    variances_list = [variance_row[c] for c in feature_column_names]
    return np.array(variances_list)


vars_ = getFeatVecs(labelsAndData_df, feature_cols)

thedata_rdd = thedata.rdd.map(lambda row: row.features.toArray())
mean = thedata_rdd.map(lambda x: x[1]).reduce(add) / n
print(thedata_rdd.filter(lambda x: x[1] > 10*mean).count())

indices_of_variance = [t[0] for t in sorted(enumerate(vars_), key=lambda x: x[1])[-3:]]

predictions = k_clusters.transform(thedata)

rdd0 = predictions.filter(col("prediction") == 0).select("features").rdd.map(lambda row: row.features.toArray())
rdd1 = predictions.filter(col("prediction") == 1).select("features").rdd.map(lambda row: row.features.toArray())

cluster_0 = rdd0.take(5)
cluster_1 = rdd1.take(5)

cluster_0_projected = np.array([[point[i] for i in indices_of_variance] for point in cluster_0])
cluster_1_projected = np.array([[point[i] for i in indices_of_variance] for point in cluster_1])

M = max(max(cluster_1_projected.flatten()), max(cluster_0_projected.flatten()))
m = min(min(cluster_1_projected.flatten()), min(cluster_0_projected.flatten()))

fig2plot = plt.figure(figsize=(8, 8))
pltx = fig2plot.add_subplot(111, projection='3d')
pltx.scatter(cluster_0_projected[:, 0], cluster_0_projected[:, 1], cluster_0_projected[:, 2], c="b")
pltx.scatter(cluster_1_projected[:, 0], cluster_1_projected[:, 1], cluster_1_projected[:, 2], c="r")
pltx.set_xlim(m, M)
pltx.set_ylim(m, M)
pltx.set_zlim(m, M)
pltx.legend(["cluster 0", "cluster 1"])
plt.show()


def euclidean_distance_points(x1, x2):
    x3 = x1 - x2
    return np.sqrt(x3.T.dot(x3))


WSSSE = k_clusters.summary.trainingCost
print("Within Set Sum of Squared Error = " + str(WSSSE))

predictions_with_labels = k_clusters.transform(df_assembled.select("label", "features"))
clusterLabel = predictions_with_labels.groupBy("prediction", "label").count()

for items in clusterLabel.collect():
    print(items)


# MÉTODO DEL CODO - DATOS SIN NORMALIZAR

k_values = range(5, 126, 20)

def clustering_error_Score(thedata_df, k):
    kmeans = KMeans(k=k, maxIter=10, initMode="random", seed=42)
    model = kmeans.fit(thedata_df)
    WSSSE = model.summary.trainingCost
    return WSSSE


k_scores = [clustering_error_Score(thedata, k) for k in k_values]
for score in k_scores:
    print(score)
    
plt.scatter(k_values, k_scores)
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Error de Clustering (WSSSE)')
plt.show()

# NORMALIZACIÓN DE DATOS

scaler = StandardScaler(inputCol="features", 
                        outputCol="scaledFeatures",
                        withStd=True,
                        withMean=True)

scalerModel = scaler.fit(thedata)

normalized = scalerModel.transform(thedata).select('scaledFeatures')
normalized = normalized.withColumnRenamed('scaledFeatures', 'features').cache()

print("Datos normalizados (2 ejemplos)")
print(normalized.take(2))

# MÉTODO DEL CODO - DATOS NORMALIZADOS

k_range = range(60, 111, 10)
k_scores = [clustering_error_Score(normalized, k) for k in k_range]

print("Valores de error")
for kscore in k_scores:
    print(kscore)

plt.plot(k_range, k_scores, marker='o', linestyle='-')
plt.xlabel("Número de Clústeres (k)")  
plt.ylabel("Error de Clustering (WSSSE)")  
plt.title("Método del Codo para Selección de k") 
plt.grid(True) 
plt.show()


# VISUALIZACIÓN ANTES DE NORMALIZACION

K_norm = 90

var = getFeatVecs(labelsAndData_df, feature_cols)
indices_of_variance = [t[0] for t in sorted(enumerate(var), key=lambda x: x[1])[-3:]]

dataprojected = thedata.randomSplit([1.0, 999.0])[0].cache()

kclusters = KMeans(k=K_norm, maxIter=10, initMode="random", seed=42).fit(thedata)

listdataprojected = dataprojected.rdd.map(lambda row: row.features.toArray()).collect()
projected_data = np.array([[point[i] for i in indices_of_variance] for point in listdataprojected])

predictions_proj = kclusters.transform(dataprojected)
klabels = [row.prediction for row in predictions_proj.collect()]

Maxi = max(projected_data.flatten())
mini = min(projected_data.flatten())

figs = plt.figure(figsize=(8, 8))
pltx = figs.add_subplot(111, projection='3d')
pltx.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], c=klabels)
pltx.set_xlim(mini, Maxi)
pltx.set_ylim(mini, Maxi)
pltx.set_zlim(mini, Maxi)
pltx.set_title("Antes de la normalización")
plt.show()

# VISUALIZACIÓN DESPUÉS DE NORMALIZACION (GRÁFICA 5)

normalized_rdd = normalized.rdd.map(lambda row: row.features.toArray().tolist())
normalized_with_cols = normalized_rdd.map(
    lambda x: Row(**{f"feature_{i}": x[i] for i in range(len(x))})
).toDF()

var_normalized = getFeatVecs(normalized_with_cols, feature_cols)
indices_of_variance_norm = [t[0] for t in sorted(enumerate(var_normalized), key=lambda x: x[1])[-3:]]

kclusters = KMeans(k=K_norm, maxIter=10, initMode="random", seed=42).fit(normalized)

dataprojected_normed = scalerModel.transform(dataprojected).select('scaledFeatures')
dataprojected_normed = dataprojected_normed.withColumnRenamed('scaledFeatures', 'features').cache()

dataprojected_normed_list = dataprojected_normed.rdd.map(lambda row: row.features.toArray()).collect()
projected_data = np.array([[point[i] for i in indices_of_variance] for point in dataprojected_normed_list])

predictions_normed = kclusters.transform(dataprojected_normed)
klabels = [row.prediction for row in predictions_normed.collect()]

Maxi = max(projected_data.flatten())
mini = min(projected_data.flatten())

figs = plt.figure(figsize=(8, 8))
pltx = figs.add_subplot(111, projection='3d')
pltx.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], c=klabels)
pltx.set_xlim(mini, Maxi)
pltx.set_ylim(mini, Maxi)
pltx.set_zlim(mini, Maxi)
pltx.set_title("Despues de la normalización")

print(f"Puntos en dataprojected: {dataprojected.count()}")

plt.show()  # Gráfica 5