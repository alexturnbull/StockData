import yfinance as yf 
from datetime import date, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os as os


today = date.today()

spark = SparkSession.builder \
        .appName('Company_Project_1.0.1') \
        .config("spark.jars", "mariadb-java-client-3.1.4.jar")\
        .getOrCreate()


def GetStockData(stock):
    #gets stock data from today back 14yrs. stock and date inputs should be strings and progress should be booleen
    end_date = today.strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=20000)).strftime("%Y-%m-%d")
    progress = False
    stock_data = yf.download(stock, start_date, end_date, progress)
    stock_data["Date"] = stock_data.index
    stock_data = stock_data [["Date", "Open", "High", "Low", "Close",  "Volume"]]
    stock_data.reset_index(drop=True, inplace=True)
    stock_data = spark.createDataFrame(stock_data)
    stock_data = stock_data.withColumn("Name", lit(stock))
    return stock_data



DOCS_DATA = GetStockData('DOCS')


DOCS_DATA.write \
    .format("jdbc") \
    .mode("overwrite") \
    .option("driver", "org.mariadb.jdbc.Driver") \
    .option("url", "jdbc:mariadb://localhost:3306/lnd") \
    .option("dbtable", 'DOCS3') \
    .option("user", "ETL") \
    .option("password", os.environ.get('PASS')) \
    .save()











