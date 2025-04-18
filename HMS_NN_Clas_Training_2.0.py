#=====NN Classifier to initially classify impedances====
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#Load data
def read_single_table():
    #Realpart 
    data_real = pd.read_csv('./HMS_Classification/Nyquist_Data_classification_Real_part_test_V10_delamination.csv', delimiter = ';', header = 0, decimal = '.')# read as pandas DataFrame 
    #print("data_real:\n",data_real)

    #Imaginarypart 
    #data_imag = pd.read_csv('./HMS_Data base/Nyquist_Data_adapted_Imag_part.csv', delimiter = ';', header = 0, decimal = '.')# read as pandas DataFrame 
    #print(data_imag)

    return data_real
data_real=read_single_table()
print("data_real:\n",data_real)


#X_Data, y_Data
#X_data_set_list=["RH_Air", "Lambda", "Temperature", "Frequency"]
X_data_set=data_real.iloc[:,:-1]
#X_data_set=pd.DataFrame(data_real_op_point[{"RH_Air", "Lambda", "Temperature", "Frequency"}]) -->Alternative 1     
print("\nType of X_data_set-Data: ",type(X_data_set))
print("\nX_data_set-Data: \n",X_data_set)
print("\nShape of X_data_set-Data: \n",X_data_set.shape)

y=data_real.iloc[:,-1]
#y=data_real_op_point.iloc[:,4] --> Alternative
#y=y.transpose()
print("\nType of y-Data: ",type(y))
print("\ny-Data: \n",y)
print("\nShape of y-Data: \n",y.shape)


# Separation in training-, test, and validation-data
X_train_full, X_test, y_train_full, y_test = train_test_split(X_data_set, y, test_size = 0.2, random_state = 20, shuffle = True) # define radnom test, training and validation data 
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state = 20, shuffle = True)# from Training data full define validation data --> detect overfit or underfit



print("\nHorizontal shape of X_data_set-Data: ",X_data_set.shape[1:])

# Define NN Classifier
tf.random.set_seed(42)
model = tf.keras.Sequential([ 
    tf.keras.layers.Flatten(input_shape=X_data_set.shape[1:]), 
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax") ])
model.summary()

optimizer = tf.keras.optimizers.SGD() #(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])


def step_decay(epoch):
    initial_lr = 0.01  # Starting learning rate
    drop = 0.5         # Factor to reduce the learning rate (percentage)
    epochs_drop = 50   # Reduce every 50 epochs
    return initial_lr * (drop ** (epoch // epochs_drop))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)

print("Learning rate:", model.optimizer.lr.numpy())


history = model.fit(X_train, y_train, epochs = 500, validation_data = (X_valid, y_valid), callbacks=[lr_scheduler])


# Show NN efficiency
pd.DataFrame(history.history).plot( figsize=(8, 5), grid= True , xlabel="Epoch", style=["r--", "b--", "g--", "y--"],ylim=[0, 2]) #, xlim=[0, 30], ylim=[0, 1]
plt.show()


# Initialize class names
class_names  = ["Healthy", "Low failure", "High failure"]
print(class_names)
print(y[0])
print(class_names[int(y[0])])
print(class_names[0])


 # Save trained model
model.save('./HMS_Training_mein_modell/mein_modell_CL_NN_TBD.h5')
print("\nTraining successful!\n")


# Predict Class
X_to_predict = pd.DataFrame(X_data_set[:1])
X_to_predict=X_to_predict.append(X_data_set.iloc[11,:])
X_to_predict=X_to_predict.append(X_data_set.iloc[22,:])
print("X_to_predict:\n", X_to_predict)

X_to_predict_classes = pd.DataFrame(data_real[:1])
X_to_predict_classes=X_to_predict_classes.append(data_real.iloc[11,:])
X_to_predict_classes=X_to_predict_classes.append(data_real.iloc[22,:])
print("X_to_predict_classes:\n", X_to_predict_classes)

y_pred = model.predict(X_to_predict)
print(y_pred.round(2))

y_pred_max = y_pred.argmax(axis = 1)
print(y_pred_max)
print("Class: ", np.array(class_names)[y_pred_max])

