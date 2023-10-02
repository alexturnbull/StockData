from datetime import date, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os as os


today = date.today()

spark = SparkSession.builder \
        .appName('Company_Project') \
        .config("spark.jars", "mariadb-java-client-3.1.4.jar")\
        .getOrCreate()

# Create DataFrame 
columns = ['id', 'name','age','gender']
data = [(1, 'James',30,"M"), (2, "Ann",40,"F"),
    (3, 'Jeff',41,'M'),(4, 'Jennifer',20,'F')]

sampleDF = spark.sparkContext.parallelize(data).toDF(columns)

sampleDF.write\
    .format("jdbc") \
    .mode("overwrite") \
    .option("driver", "org.mariadb.jdbc.Driver") \
    .option("url", "jdbc:mariadb://localhost:3306/lnd") \
    .option("dbtable", 'test') \
    .option("user", "ETL") \
    .option("password", os.environ.get('PASS')) \
    .save()
