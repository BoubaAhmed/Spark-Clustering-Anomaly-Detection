spark.sql("""
CREATE EXTERNAL TABLE IF NOT EXISTS sensor_data_clean (
    date STRING,
    time STRING,
    T1 FLOAT,
    T2 FLOAT,
    T3 FLOAT,
    T4 FLOAT,
    T5 FLOAT,
    T6 FLOAT,
    T7 FLOAT,
    T8p1 FLOAT,
    T8p2 FLOAT,
    TAbluft FLOAT,
    TZuluft FLOAT,
    Volumenstrom FLOAT,
    Zuluft FLOAT,
    Abluft FLOAT,
    Foliendicke FLOAT,
    Fuellstand FLOAT,
    Geschwindigkeit FLOAT,
    Innendruck FLOAT,
    Rakelhoehe FLOAT,
    Ventilstellung FLOAT,
    original_file STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ';'
STORED AS TEXTFILE
LOCATION 'hdfs://localhost:9000/spark_clustering/cleaned_data'
""")