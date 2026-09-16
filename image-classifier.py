import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Dropout
#LOAD CIPHAR
(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

print(f"Training data:{x_train.shape},{y_train.shape}")
print(f"Testing data:{x_test.shape},{y_test.shape}")

model=Sequential(
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3))
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPool2D((2,2))
)