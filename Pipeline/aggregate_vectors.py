from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, avg
from pyspark.sql.types import FloatType

spark = SparkSession.builder.appName("AggregateVectors").getOrCreate()

# Lire les données nettoyées avec schéma explicite
df_clean = spark.read \
    .option("sep", ";") \
    .option("header", "true") \
    .csv("hdfs://localhost:9000/spark_clustering/cleaned_data/*.csv")

# Liste des colonnes à convertir en float (ajuster selon vos données)
numeric_cols = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7",
    "T8p1", "T8p2", "TAbluft", "TZuluft", "Volumenstrom",
    "Zuluft", "Abluft", "Foliendicke", "Fuellstand",
    "Geschwindigkeit", "Innendruck", "Rakelhoehe", "Ventilstellung"
]

# Conversion finale des types numériques
for col_name in numeric_cols:
    df_clean = df_clean.withColumn(col_name, df_clean[col_name].cast(FloatType()))

# Ajout du nom de fichier et filtrage des colonnes
df_with_filename = df_clean.withColumn("filename", input_file_name()) \
    .select(["filename"] + numeric_cols)

# Calcul des moyennes
aggregated = df_with_filename.groupBy("filename").agg(
    *[avg(col).alias(f"mean_{col}") for col in numeric_cols]
)

# Écriture avec format cohérent
aggregated.write \
    .option("sep", ";") \
    .option("header", "true") \
    .mode("overwrite") \
    .csv("hdfs://localhost:9000/spark_clustering/aggregated_vectors")

spark.stop()