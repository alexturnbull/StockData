import tensorflow as tf 
import numpy as np 
import matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt 
import os as os
import pandas as pd
from ML_tools import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import DateType
from sklearn.preprocessing import StandardScaler


spark = SparkSession.builder \
        .appName('Company_Project') \
        .config("spark.jars", "mariadb-java-client-3.1.4.jar")\
        .getOrCreate()

DOCS = spark.read.format("jdbc")\
        .option("url","jdbc:mariadb://localhost:3306/lnd")\
        .option("driver", "org.mariadb.jdbc.Driver")\
        .option("dbtable", "DOCS")\
        .option("user", "ETL")\
        .option("password", os.environ.get('PASS'))\
        .load()


# addiing previsu days closing proce to row
w = Window().partitionBy().orderBy(col('Date'))
DOCS_ML = DOCS.withColumn("PrevClose", lag("close", 1, 0).over(w)) \
        .withColumn("Return", (col("close") - col("PrevClose"))/ col("PrevClose")) \
        .withColumn("Date", DOCS.Date.cast(DateType()))


DOCS_pd = DOCS_ML.toPandas()

# take return into a 1 dimentional numpy ary
series = DOCS_pd['Return'].values[1:].reshape(-1, 1)
print(series)
# normalise data
scaler = StandardScaler()
scaler.fit(series[:len(series // 2)])
series = scaler.transform(series).flatten()

print(series)

# Setup the data sets 

# N = number of samples in the data set
# T = Sequence length
# D =number of featurs
# M = number of hidden units
# K = number of output units

T = 10 
# T is the number of previous steps we will use to calculate the next 
D = 1
# D is 1 as we are only using 1 feature in this case Return 
X = []
Y = [] 

for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, 1) #now data should be N x T x D
Y = np.array(Y)
N = len(X)

# this sets x as our inputs and y as our target 

i = tf.keras.layers.Input(shape=(T,1))
x = tf.keras.layers.LSTM(5)(i)
o = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(i, o)

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))

r = model.fit(X[:-N//2], Y[:-N//2], 
              epochs = 90,
              validation_data=(X[-N//2:], Y[N//2:]),
              )



# multi step forcast 
validation_target = Y[N//2:]
validation_predictions = []

# last train input 
last_x = X[-N//2]

while len (validation_predictions) < len(validation_target):
#     1x1 array -> scalar
    p = model.predict(last_x.reshape(1, T, 1))[0,0]
    # update the prediction list
    validation_predictions.append(p)
    # make the new input 
    last_x = np.roll(last_x, -1)
    last_x[-1] = p 



plt.plot(validation_target, label='forcast target')
plt.plot(validation_predictions, label='forcast prediction')
plt.legend()
plt.show()
        




