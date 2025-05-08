from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, avg
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
import pickle
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamingJob")

spark = SparkSession.builder \
    .appName("RealTimeAnomalyDetection") \
    .config("spark.sql.streaming.checkpointLocation", "hdfs://localhost:9000/spark_clustering/checkpoints") \
    .getOrCreate()

# ----------------------------------------------------------
# Correction du schéma pour correspondre à la structure des fichiers
# ----------------------------------------------------------
numeric_cols = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", 
    "T8p1", "T8p2", "TAbluft", "TZuluft", "Volumenstrom",
    "Zuluft", "Abluft", "Foliendicke", "Fuellstand",
    "Geschwindigkeit", "Innendruck", "Rakelhoehe", "Ventilstellung"
]

schema = StructType(
    [StructField(c, FloatType()) for c in numeric_cols] +
    [
        StructField("date", StringType()),
        StructField("time", StringType()),
        StructField("original_file", StringType())
    ]
)

# ----------------------------------------------------------
# Chargement des modèles depuis fichiers locaux et HDFS
# ----------------------------------------------------------
try:
    # Chargement du modèle DBSCAN depuis le disque local
    with open("dbscan_model.pkl", "rb") as f:
        dbscan_model = pickle.load(f)
        
    # Chargement du modèle StandardScaler depuis HDFS
    scaler_model = StandardScalerModel.load("hdfs://localhost:9000/spark_clustering/models/scaler_model")
    
    logger.info("Modèles chargés avec succès")
except Exception as e:
    logger.error(f"Erreur de chargement des modèles: {str(e)}")
    raise


# ----------------------------------------------------------
# Fonction de traitement améliorée
# ----------------------------------------------------------
def process_batch(batch_df, batch_id):
    try:
        logger.info(f"Début du traitement du batch {batch_id}")
        
        # 1. Nettoyage des données
        cleaned = batch_df.dropna()
        if cleaned.isEmpty():
            logger.info(f"Batch {batch_id} vide après nettoyage")
            return

        # 2. Calcul des moyennes
        aggregated = cleaned.groupBy("original_file").agg(
            *[avg(col(c)).alias(f"mean_{c}") for c in numeric_cols]
        )
        
        # 3. Préparation des features
        assembler = VectorAssembler(
            inputCols=[f"mean_{c}" for c in numeric_cols],
            outputCol="raw_features"
        )
        features = assembler.transform(aggregated)
        
        # 4. Normalisation
        scaled = scaler_model.transform(features)
        
        # 5. Conversion en numpy array
        features_list = [row["scaled_features"].toArray() for row in scaled.collect()]
        features_array = np.array(features_list)
        
        # 6. Détection d'anomalies
        clusters = dbscan_model.fit_predict(features_array)
        
        # 7. Création des résultats
        results = [
            (row["original_file"], int(cluster), bool(cluster == -1), batch_id)
            for row, cluster in zip(scaled.collect(), clusters)
        ]
        
        results_df = spark.createDataFrame(
            results, 
            ["original_file", "cluster", "is_anomaly", "batch_id"]
        )
        
        # 8. Vérification des anomalies dans les résultats
        anomalies = results_df.filter(col("is_anomaly") == True).count()
        
        if anomalies > 0:
            # Si des anomalies sont détectées, écriture des résultats
            output_path = "hdfs://localhost:9000/spark_clustering/stream_results"
            (results_df
             .withColumn("processing_time", current_timestamp())
             .write
             .option("sep", ";")
             .option("header", "true")
             .mode("append")
             .csv(output_path))
            logger.info(f"Batch {batch_id} traité avec succès. Résultats sauvegardés dans {output_path}")
        else:
            logger.info(f"Batch {batch_id} traité sans anomalies. Aucun résultat sauvegardé.")

    except Exception as e:
        logger.error(f"Échec du traitement du batch {batch_id}: {str(e)}", exc_info=True)
        raise

# ----------------------------------------------------------
# Configuration du flux de streaming
# ----------------------------------------------------------
streaming_query = (
    spark.readStream
    .schema(schema)
    .option("sep", ";")
    .option("header", "true")
    .option("enforceSchema", "false")
    .option("maxFilesPerTrigger", 1)
    .option("recursiveFileLookup", "true")  # Enables recursive search through folders
    .csv("hdfs://localhost:9000/spark_clustering/stream_input/")
    .writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation", "hdfs://localhost:9000/spark_clustering/checkpoints")
    .start()
)


# Surveillance en temps réel
try:
    while True:
        logger.info(f"Statut du streaming: {streaming_query.status}")
        logger.info(f"Dernière progression: {streaming_query.lastProgress}")
        time.sleep(10)
except KeyboardInterrupt:
    logger.info("Arrêt demandé...")
    streaming_query.stop()
    spark.stop()