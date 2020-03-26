#Подключение модулей
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

def logic(a, b, c):
    return ((a != b) and (b != c))

def each_element(weights, dataset):
    temp_dataset = dataset.copy()
    length_dataset = len(temp_dataset)
    length_weights = len(weights)
    activation = [relu for i in range(length_weights-1)]
    activation.append(sigmoid)
    for w in range(length_weights):
        length_w = len(weights[w][1])
        res = np.zeros((length_dataset, length_w))
        for i in range(length_dataset):
            for j in range(length_w):
                sum = 0
                for k in range(len(dataset[i])):
                    sum += temp_dataset[i][k] * weights[w][0][k][j]
                res[i][j] = activation[w](sum + weights[w][1][j])
        temp_dataset = res
    return temp_dataset

def tensors(weights, dataset):
    dataset_temp = dataset.copy()
    length_weights = len(weights)
    activation = [relu for i in range(length_weights - 1)]
    activation.append(sigmoid)
    for w in range(length_weights):
        dataset_temp = activation[w](np.dot(dataset_temp, weights[w][0]) + weights[w][1])
    return dataset_temp

def relu(x):
    return np.maximum(x, 0.)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    

dataset_x = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]])
dataset_y = np.array([int(logic(*i)) for i in dataset_x])
print(dataset_y)
#создание модели
model = Sequential()
model.add(Dense(9, activation='relu', input_shape=(3,)))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#получение весов
weights = [layer.get_weights() for layer in model.layers]

#прогон без обучения
model_predict = model.predict(dataset_x)
each_predict = each_element(weights,dataset_x)
tensor_predict = tensors(weights, dataset_x)

print("each element predict: \n", each_predict)
print("tensors predict: \n", tensor_predict)
print("model predict: \n", model_predict)

#обучение
model.fit(dataset_x, dataset_y, epochs=260, batch_size=1, validation_split=0)

#получение весов
weights = [layer.get_weights() for layer in model.layers]

#прогон после обучения
model_predict = model.predict(dataset_x)
each_predict = each_element(weights,dataset_x)
tensor_predict = tensors(weights, dataset_x)

print("each element predict: \n", each_predict)
print("tensors predict: \n", tensor_predict)
print("model predict: \n", model_predict)