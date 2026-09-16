import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
#LOAD CIPHAR
(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0

y_train=to_categorical(Y_train,10)
y_test=to_categorical(Y_test,10)

print(f"Training data:{X_train.shape},{y_train.shape}")
print(f"Testing data:{X_test.shape},{y_test.shape}")

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    MaxPool2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#train the model
history=model.fit(X_train,Y_train,validation_split=0.2,
                  epochs=5,batch_size=64,verbose=1)

#evlaute the model
loss,accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(f"Baseline model Accuracy:{accuracy:.4f}")
