from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import logging
import matplotlib.pyplot as plt
from kneed import KneeLocator
import os
import pickle


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Clustering")

def main():
    # Initialisation de Spark avec des options de mémoire
    spark = SparkSession.builder \
        .appName("AnomalyDetectionDBSCAN") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    try:
        # 1. Lecture des fichiers CSV depuis HDFS
        logger.info("Lecture des données agrégées...")
        df = spark.read.option("sep", ";").option("header", True) \
            .csv("hdfs://localhost:9000/spark_clustering/aggregated_vectors/*.csv")

        # 2. Conversion des colonnes mean_* en float
        feature_columns = [c for c in df.columns if c.startswith("mean_")]
        for col_name in feature_columns:
            df = df.withColumn(col_name, col(col_name).cast(FloatType()))

        # 3. Création des vecteurs de caractéristiques
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
        df_vectors = assembler.transform(df)

        # 4. Normalisation des vecteurs (StandardScaler)
        scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features")
        scaler_model = scaler.fit(df_vectors)
        df_scaled = scaler_model.transform(df_vectors)

        # 5. Transformation des vecteurs PySpark en tableau numpy
        logger.info("Préparation des données pour DBSCAN...")
        vector_to_array = udf(lambda v: v.toArray().tolist(), "array<double>")
        df_array = df_scaled.withColumn("features_array", vector_to_array("scaled_features"))

        # 6. Collecte des données sous forme (filename, features)
        data = df_array.select("filename", "features_array").collect()
        filenames = [row["filename"] for row in data]
        features = np.array([row["features_array"] for row in data])

        # 7. Application de DBSCAN (clustering)
        logger.info("Application du clustering DBSCAN...")

        # 7.1 Calcul des distances vers le k-ième voisin
        k = 3  # min_samples - 1
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(features)
        distances, indices = neighbors_fit.kneighbors(features)
        k_distances = np.sort(distances[:, k - 1])  # distances triées

        # 7.2 Détection automatique du "coude" (knee) pour choisir eps
        kneedle = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
        optimal_eps = k_distances[kneedle.knee] if kneedle.knee is not None else 0.5

        logger.info(f"Valeur optimale estimée pour eps : {optimal_eps}")
        logger.info(f"k_distances : {k_distances}")
        logger.info(f"kneedle : {kneedle.knee}")

        # 7.3 Exécution de DBSCAN avec eps détecté automatiquement
        dbscan = DBSCAN(eps=optimal_eps, min_samples=4)
        clusters = dbscan.fit_predict(features)


        logger.info("Sauvegarde des modèles dans HDFS...")
        
        # 1. Sauvegarde du modèle StandardScaler
        scaler_model.write().overwrite().save("hdfs://localhost:9000/spark_clustering/models/scaler_model")

        # Save the DBSCAN model to a file
        with open('dbscan_model.pkl', 'wb') as file:
            pickle.dump(dbscan, file)

        # hdfs dfs -put dbscan_model.pkl /spark_clustering/models/
        
        logger.info(f"Modèles sauvegardés dans hdfs://localhost:9000/spark_clustering/models/")



        # 8. Création du DataFrame résultat avec cluster et anomalie
        result_data = [(filename, int(cluster), bool(cluster == -1)) for filename, cluster in zip(filenames, clusters)]
        result_df = spark.createDataFrame(result_data, ["filename", "cluster", "is_anomaly"])

        # 9. Sauvegarde des résultats dans HDFS
        logger.info("Sauvegarde des résultats dans HDFS...")
        result_df.coalesce(1) \
            .write \
            .option("header", "true") \
            .option("sep", ";") \
            .mode("overwrite") \
            .csv("hdfs://localhost:9000/spark_clustering/clustering_results")

        # 10. Logging du nombre d’anomalies trouvées
        anomaly_count = result_df.filter("is_anomaly = true").count()
        logger.info(f"Détection terminée. Nombre d'anomalies détectées : {anomaly_count}")

        # 11. Visualisation (facultatif mais utile pour debug ou rapport)
        
        # 11.1 Tracer distances vers 3ème plus proche voisin (ex: méthode alternative)
        neighbors = NearestNeighbors(n_neighbors=3)
        neighbors_fit = neighbors.fit(features)
        distances, indices = neighbors_fit.kneighbors(features)


        # 11.2 Tracer les distances utilisées pour détecter eps (officiel)
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        if kneedle.knee is not None:
            plt.axvline(x=kneedle.knee, color='r', linestyle='--', label='Coude détecté')
        plt.xlabel("Points triés")
        plt.ylabel(f"Distance au {k}e plus proche voisin")
        plt.title("Méthode du coude pour choisir eps automatiquement")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/eps_selection.png")
        plt.close()

    except Exception as e:
        logger.error(f"Erreur lors du traitement : {str(e)}", exc_info=True)
        raise
    finally:
        # Fermeture de la session Spark
        spark.stop()

if __name__ == "__main__":
    main()
