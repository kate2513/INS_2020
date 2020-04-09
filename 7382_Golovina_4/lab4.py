from keras.datasets import mnist
from keras import optimizers
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from PIL import Image
import numpy as np

def getImage(filename):
    img = Image.open(filename)
    img = img.resize((28,28))
    width, height = img.size
    array_img = np.array(img.getdata(), dtype='uint8')
    img.close()
    if array_img.ndim > 1:
      array_img = array_img[:,0]  
    array_img = np.reshape(array_img,(width,height))
    return array_img

def build_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(784, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    #opt = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer="adam",loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    

def main():
    #Загрузка данных
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    #построение модели
    model = build_model()

    #обучение
    model.fit(train_images, train_labels, epochs=5, batch_size=128)

    #оценка
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc,'\ntest_loss:', test_loss)
'''
    #предсказание своего изображения
    image = np.array([getImage('ex2.png')])
    print(model.predict(image))
'''
main()