from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, input_file_name, regexp_extract

spark = SparkSession.builder.appName("DataCleaning").getOrCreate()

# Lire les données depuis HDFS avec le nom du fichier original
df = spark.read \
    .option("sep", ";") \
    .option("header", "true") \
    .csv("hdfs://localhost:9000/spark_clustering/input_csv/*.csv") \
    .withColumn("original_file", input_file_name()) \
    .drop("_c0") \
    .drop("_c1")

# Extraire le nom simple du fichier (optionnel)
df = df.withColumn("original_file",
    regexp_extract(col("original_file"), "/([^/]+\.csv)$", 1))  # Ex: "capteur_23.csv"

# Liste des colonnes numériques (identique à votre version)
numeric_cols = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", 
    "T8p1", "T8p2", "TAbluft", "TZuluft", "Volumenstrom",
    "Zuluft", "Abluft", "Foliendicke", "Fuellstand",
    "Geschwindigkeit", "Innendruck", "Rakelhoehe", "Ventilstellung"
]

# Nettoyage des données (identique)
for col_name in numeric_cols:
    df = df.withColumn(col_name, regexp_replace(col(col_name), ",", ".").cast("float"))

# Étape 1: Supprimer les lignes avec valeurs manquantes
df_clean = df.na.drop()

# Étape 2: Calcul dynamique des plages valides
def get_dynamic_ranges(df, numeric_cols, lower_percentile=0.01, upper_percentile=0.99):
    ranges = {}
    for col_name in numeric_cols:
        quantiles = df.approxQuantile(col_name, [lower_percentile, upper_percentile], 0.01)
        # Marge de sécurité de 10% pour éviter les cas limites
        margin = (quantiles[1] - quantiles[0]) * 0.10
        ranges[col_name] = (
            max(quantiles[0] - margin, -float('inf')),  # Borne inférieure réaliste
            min(quantiles[1] + margin, float('inf'))    # Borne supérieure réaliste
        )
    return ranges

valid_ranges = get_dynamic_ranges(df_clean, numeric_cols)

# Étape 3: Filtrage des valeurs extrêmes
for col_name, (min_val, max_val) in valid_ranges.items():
    df_clean = df_clean.filter(col(col_name).between(min_val, max_val))


# Écrire les données nettoyées AVEC le nom du fichier original
df_clean.repartition(23) \
    .write \
    .option("sep", ";") \
    .option("header", "true") \
    .mode("overwrite") \
    .csv("hdfs://localhost:9000/spark_clustering/cleaned_data")

spark.stop()