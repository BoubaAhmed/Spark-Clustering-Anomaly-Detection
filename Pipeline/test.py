from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType, StringType
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
import logging
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Clustering")

def main():
    # Initialize Spark with Arrow disabled and memory configuration
    spark = SparkSession.builder \
        .appName("AnomalyDetectionDBSCAN") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    try:
        # 1. Read and prepare data
        logger.info("Reading aggregated data...")
        df = spark.read.option("sep", ";").option("header", True) \
            .csv("hdfs://localhost:9000/spark_clustering/aggregated_vectors/*.csv")

        # Convert numeric columns
        feature_columns = [c for c in df.columns if c.startswith("mean_")]
        for col_name in feature_columns:
            df = df.withColumn(col_name, col(col_name).cast(FloatType()))

        # 2. Feature engineering
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
        df_vectors = assembler.transform(df)

        scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features")
        scaler_model = scaler.fit(df_vectors)
        df_scaled = scaler_model.transform(df_vectors)

        # 3. Convert vectors to array for DBSCAN
        logger.info("Preparing data for DBSCAN...")
        
        # UDF to convert vector to numpy array
        vector_to_array = udf(lambda v: v.toArray().tolist(), "array<double>")
        df_array = df_scaled.withColumn("features_array", vector_to_array("scaled_features"))
        
        # Collect data with filename and features
        data = df_array.select("filename", "features_array").collect()
        filenames = [row["filename"] for row in data]
        features = np.array([row["features_array"] for row in data])

        # 4. Apply DBSCAN
        logger.info("Applying DBSCAN clustering...")
        dbscan = DBSCAN(eps=3.1, min_samples=2)  # Adjusted parameters
        clusters = dbscan.fit_predict(features)
        
        # Create result DataFrame
        result_data = [(filename, int(cluster), bool(cluster == -1)) for filename, cluster in zip(filenames, clusters)]
        result_df = spark.createDataFrame(result_data, ["filename", "cluster", "is_anomaly"])

        # 5. Save results
        logger.info("Saving results to HDFS...")
        
        # Convert cluster info to string for CSV output
        result_df.coalesce(1) \
            .write \
            .option("header", "true") \
            .option("sep", ";") \
            .mode("overwrite") \
            .csv("hdfs://localhost:9000/spark_clustering/clustering_results")

        # Log results
        anomaly_count = result_df.filter("is_anomaly = true").count()
        logger.info(f"Detection complete. Found {anomaly_count} anomalous files.")

        # Calcul des distances vers les k (ex: 3) plus proches voisins
        neighbors = NearestNeighbors(n_neighbors=3)
        neighbors_fit = neighbors.fit(features)
        distances, indices = neighbors_fit.kneighbors(features)

        # Trier les distances vers le 3ème voisin (indice 2 car zéro-indexé)
        distances = np.sort(distances[:, 2])  

        # Tracer la courbe pour visualiser le "knee"
        plt.plot(distances)
        plt.xlabel("Points triés")
        plt.ylabel("Distance au 3e plus proche voisin")
        plt.title("Méthode du coude pour choisir eps")
        plt.grid(True)
        plt.show()


    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()