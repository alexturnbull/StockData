import tensorflow as tf 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from ML_tools import *


#Genrate test data 
series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1
plt.plot(series)
plt.show()


#build the data set 

T = 10 
X = [ ]
Y = [ ]
for t in range(len(series) - T): 
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print('x_shape' , X.shape, 'y_shape', Y.shape)

#How to under stand the recursion of an RNN https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9


i = tf.keras.layers.Input(shape=(T,1))
x = tf.keras.layers.SimpleRNN(8, activation='tanh')(i)
o = tf.keras.layers.Dense(1)(x) 

model = tf.keras.models.Model(i, o)

model.compile(
              loss='mse',
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

#train model
r = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs=80,
    validation_data = ([X[-N//2:]], Y[-N//2:]),

)


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
break_clause()
 
#this would be theright wat

val_target = Y[-N//2: ]
val_prediction = []

   # index on the last training input 
last_x = X[-N//2]


while len(val_prediction) < len(val_target): 
  p = model.predict(last_x.reshape(1, -1))[0,0]
  val_prediction.append(p)
  last_x = np.roll(last_x, -1) #this shifts everything to the left and updats the last valrible with out new prediction
  last_x[-1] = p



plt.plot(val_prediction, label= 'Forcast')
plt.plot(val_target, label = 'act')
plt.legend()
plt.show()

save_model()