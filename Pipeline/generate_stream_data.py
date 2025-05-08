from pyspark.sql import SparkSession
import numpy as np
import time
from datetime import datetime
from pyspark.sql.functions import col, lit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataGenerator")

spark = SparkSession.builder.appName("DataGenerator").getOrCreate()

numeric_cols = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7",
    "T8p1", "T8p2", "TAbluft", "TZuluft", "Volumenstrom",
    "Zuluft", "Abluft", "Foliendicke", "Fuellstand",
    "Geschwindigkeit", "Innendruck", "Rakelhoehe", "Ventilstellung"
]

try:
    df_base = spark.read \
        .option("sep", ";") \
        .option("header", "true") \
        .csv("hdfs://localhost:9000/spark_clustering/cleaned_data/*.csv") \
        .drop("original_file")

    for c in numeric_cols:
        df_base = df_base.withColumn(c, col(c).cast("float"))
    
    df_base.persist()

    while True:
        batch_time = datetime.now()
        current_date = batch_time.strftime("%d.%m.%Y")
        current_time = batch_time.strftime("%H:%M:%S")

        for sensor_id in range(23):
            try:
                df_noisy = df_base
                for col_name in numeric_cols:
                    noise = np.random.normal(0, 0.02)
                    df_noisy = df_noisy.withColumn(col_name, col(col_name) * (1 + noise))
                    

                df_noisy = df_noisy \
                    .withColumn("date", lit(current_date)) \
                    .withColumn("time", lit(current_time)) \
                    .withColumn("original_file", lit("sensor_" + str(sensor_id)))

                output_path = f"hdfs://localhost:9000/spark_clustering/stream_input/{batch_time.timestamp()}/sensor_{sensor_id}"
                
                df_noisy.write \
                    .option("sep", ";") \
                    .option("header", "true") \
                    .mode("overwrite") \
                    .csv(output_path)

            except Exception as e:
                logger.error(f"Erreur capteur {sensor_id}: {str(e)}")
        
        time.sleep(60)

except KeyboardInterrupt:
    logger.info("Arrêt propre")
finally:
    df_base.unpersist()
    spark.stop()
