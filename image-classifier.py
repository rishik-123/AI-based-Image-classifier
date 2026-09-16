import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

#LOAD CIPHAR
(X_train,Y_train),(X_test,Y_test)=cifar10