#import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
print (tf.__version__)
# start defince the training dataset and testing dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test) = mnist.load_data()

# end of init dataset 
# the features of train_data : the pixel of image of these digit from 1->9
print (x_train[0])
#normalize the dataset
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

# print the training sets one more time

print (x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

# let to define the model for keras
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])
			  
# Once we need to start fitting the model 
model.fit(x_train,y_train,epochs=3)
val_loss, val_acc = model.evaluate(x_test,y_test)
print ('the loss value {}',val_loss)
print ('the accuracy {}',val_acc)
#finally can save the model
model.save('keras.h5')
#load it back
new_model = tf.keras.models.load_model('keras.h5')

predictions = new_model.predict(x_test)
print(predictions)
print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
 