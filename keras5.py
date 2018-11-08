import keras
import numpy as np
from keras.datasets import reuters
from keras.utils import to_categorical
from keras import models
from keras import layers

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=1000)

def to_vectorize(sequences,dimension=1000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

x_train=to_vectorize(train_data)
x_test=to_vectorize(test_data)

model=models.Sequential()
model.add(layers.Dense(64,input_shape=(1000,)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history=model.fit(x_train,train_labels,epochs=15,batch_size=512)

train_loss,train_acc=model.evaluate(x_test,test_labels)

print("Training Loss",train_loss)
print("Training accuracy",train_acc)

from keras import backend as K
K.clear_session()    
