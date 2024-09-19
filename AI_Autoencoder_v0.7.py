# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:54:28 2024

@author: Cosmin
"""
import tensorflow as tf
import numpy
import time
import xlwt 
from numpy import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler 
from keras.callbacks import History 
from sklearn.preprocessing import StandardScaler


        #Consider using noise implementation resulting in better performances
def add_noise(X, noise_factor=0.2):
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    X_noisy = X + noise
    X_noisy = np.clip(X_noisy, 0., 1.)  # Clip values to be in the same range as the original data
    return X_noisy


        #Normalize the data with any scaler you want
def normalize(dataset): 
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataset)

       #StandardNormalize
   
    #def normalize(dataset):
        #    scaler = StandardScaler()
        #    return scaler.fit_transform(dataset)

        #features & labels can be sepparate from the dataset and you can work with them separately or with both 
df=pd.read_excel(r'C:\Space\work\AI\data\DateSortate\FundeniATFull_cleaned.xlsx')
        #df2=pd.read_excel(r'vezi excel v3')

        #range of parameters
train_set=df.loc[:, 'T0m':'Dp6']
train_y=df.loc[:, 'T0m':'Dp6']


        #code convert the train_set and train_y DataFrames into NumPy arrays
X=train_set.to_numpy()
y=train_y.to_numpy();
        #split into samples
        #min() returns the min from each column is not specified the column
        #max() return the max from each column
        #for global values apply the function again on the output
minim=min(train_set)
maxim=max(train_set)

        #data normalization
X_normalized = normalize(X)

        #number of samples for training
no_test_inputs=10000


        #randomize the input features
X_normalized,y=shuffle(X_normalized,y)
        #se creaza setul de antrenare, fara ultimele no_test_inputs randuri care vor fi folosite ulterior pentru testare
        #deci se vor crea X care va avea [-no_test_inputs] randuri + X_test care va contine [no_test_inputs]

X_train, X_test = numpy.split(X_normalized, [-no_test_inputs])
y_train, y_test = numpy.split(y, [-no_test_inputs])

        #X,X_test=numpy.split(X,[-no_test_inputs])
        #y,y_test=numpy.split(y,[-no_test_inputs])


    # =============================================================================
    # ffd_layer1=tf.keras.layers.Dense(12,
    #                                     activation='elu',
    #                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(input1)
    # ffd_layer2=tf.keras.layers.Dense(12,
    #                                     activation='elu',
    #                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(ffd_layer1)
    # ffd_layer3=tf.keras.layers.Dense(12,
    #                                     activation='elu',
    #                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(ffd_layer2)
    # 
    # =============================================================================


        #input layer - 35 = dimension rang (usually match the number of parameters)
input1=tf.keras.layers.Input( shape=(35,) )

        # add Gaussian noise (0.1 is the standard deviation of the noise)

noisy_input = tf.keras.layers.GaussianNoise(0.1)(input1)

        #autoencoder structure 
        #encoder
dense1_layer1 = tf.keras.layers.Dense(35, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(noisy_input)

        #dropout
dropout1 = tf.keras.layers.Dropout(0.2)(dense1_layer1)  # Drop 20% of the neurons

dense1_layer2 = tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(dropout1)
        #in v0.4 sunt 15 neuroni pe stratul 3 - s-a schimbat in 10 pentru a incuraja retinerea patternurilor importante.

dropout2 = tf.keras.layers.Dropout(0.2)(dense1_layer2)

dense1_layer3 = tf.keras.layers.Dense(15, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(dropout2)


    
        #Decoder
dense1_layer4 = tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(dense1_layer3)

dropout3 = tf.keras.layers.Dropout(0.2)(dense1_layer4)

dense1_layer5 = tf.keras.layers.Dense(35, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(dropout3)
    
        #final output layer = usually has to match the original input dimensions
out_layer = tf.keras.layers.Dense(35, activation='linear', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0))(dense1_layer5)
        #check the shape of the output tensor of your model's last layer,
        #or a way to confirm that the output layer has the desired shape (ca primu)
out_layer.shape
    
        #model definition
autoencoder=tf.keras.Model( inputs=[input1], outputs=out_layer)


history=History()
callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=600,restore_best_weights = True)


        #a fost schimbat loss function din mean_absolute_error in mean_squared_error in v0.3 = pentru perfomante mai bune
autoencoder.compile(optimizer='Nadam', loss='mean_squared_error',metrics=['mean_squared_logarithmic_error','mae','RootMeanSquaredError' ])

        # Add noise to your training data
X_train_noisy = add_noise(X_train, noise_factor=0.05)

t1 = time.time()
        #model monitor live 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

        #1st try 
        #considera modificarea batch_size-ului pentru comportament
autoencoder.fit(X_train_noisy,y_train,verbose=1,epochs=550,validation_split=0.3,batch_size=64,callbacks=[callback, history])

        #2nd try
        #autoencoder.fit([X_train],y_train,verbose=1,epochs=1000,validation_split=0.2,batch_size=64,callbacks=[callback, history])    
t2 = time.time()


print('Time elapsed: ',t2-t1)

        #plotting loss

"""
This plots the training and validation loss after epoch 30, which can help visualize the convergence of the model. 
Starting from epoch 30 might give a clearer picture of when the model stabilized.
"""
plt.figure(2)
plt.plot(autoencoder.history.history['val_loss'][30:])
plt.plot(autoencoder.history.history['loss'][30:])

msle_train = autoencoder.history.history['mean_squared_logarithmic_error']
msle_validation = autoencoder.history.history['val_mean_squared_logarithmic_error']
loss_train=autoencoder.history.history['loss']
loss_validation=autoencoder.history.history['val_loss']


index_val=loss_validation.index(min(loss_validation))
print('MSLE train: ', msle_train[index_val])
print('MSLE validation: ', msle_validation[index_val])
print('Loss train: ',loss_train[index_val])
print('Loss validation: ',loss_validation[index_val])
print('MAE train: ',autoencoder.history.history['mae'][index_val])
print('MAE validation: ',autoencoder.history.history['val_mae'][index_val])
print('RMSE train: ',autoencoder.history.history['RootMeanSquaredError'][index_val])
print('RMSE validation: ',autoencoder.history.history['val_RootMeanSquaredError'][index_val])

        #Test evaluation

autoencoder.evaluate([X_test],y_test,verbose=2)

        #Test Example & prediction

forecast_test=autoencoder.predict([X_test])


        #Manual implementation of MSE
            #print('MSE test:', mse_test)
            #mse_test=numpy.mean(numpy.multiply(forecast_test-y_test,forecast_test-y_test))
            #Automatic implementation of MSE

mse_test = tf.keras.losses.MeanSquaredError()(y_test, forecast_test).numpy()
print('MSE test:',mse_test)


# If you need to recover the original scale X_test => y_test
#decoded_data = scaler.inverse_transform(encoded_data)


# Testare cu date exploatare 
# real_sample = X_test[0].copy()

# corrupt the sample by replacing one or more values with NaN or another value

#corrupted_sample = real_sample.copy()
#corrupted_sample[0] = 0  # Corrupt the 6th value, as an example
#corrupted_sample = corrupted_sample.reshape(1, 35) 

# SAU

#corrupted_sample = corrupted_sample.reshape(1, -1)


# Alternatively, use random noise
# corrupted_sample[5] = np.random.uniform(0, 100)

#reconstructed_sample = autoencoder.predict(corrupted_sample)






