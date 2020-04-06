from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt
import csv
import collections

def generateNumber(mean,std):
    std = np.sqrt(std)
    x = np.random.normal(mean,std)
    return x

def generateData(nrows, ncols):
    data = np.zeros((nrows, ncols))
    target = np.zeros(nrows)
    #target = np.zeros((nrows,1))
    for i in range(nrows):
        x = generateNumber(3,10)
        e = generateNumber(0,0.3)
        #print(x,' ',e)
        data[i,:] = (x ** 2 + e, np.sin(x/2) + e, np.cos(2*x) + e, x - 3 + e, -x + e, (x ** 3)/4 + e)
        target[i] = np.abs(x) + e
    return (data, target)    


def writeAsCSV(file_name, fields, mode):
    file = open(file_name, 'w')
    out = csv.writer(file, delimiter=',')
    if mode == 1:
        for item in fields:
            out.writerow(item)
    else:
        out.writerows(map(lambda x: [x], fields))
    file.close()

#генерация данных
(train_data, train_target) = generateData(200,6)
(test_data, test_target) = generateData(30,6)

#нормализация
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std
test_data -= mean
test_data /= std

#создание
encoding_dim = 2
main_input = Input(shape=(6,), name="main_input")
encoded = Dense(60, activation='relu')(main_input)
encoded = Dense(60, activation='relu')(encoded)
encoded = Dense(30, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='linear')(encoded)

decoded = Dense(60, activation='relu', name='dec1')(encoded)
decoded = Dense(60, activation='relu', name='dec2')(decoded)
decoded = Dense(60, activation='relu', name='dec3')(decoded)
decoded = Dense(6, name='dec4')(decoded)

regres = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
regres = Dense(64, activation='relu')(regres)
regres = Dense(64, activation='relu')(regres)
regres = Dense(1, name="out_regres")(regres)


#модели
autoencoder = Model(main_input, decoded)
encoded = Model(main_input, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder = autoencoder.get_layer('dec1')(encoded_input)
decoder = autoencoder.get_layer('dec2')(decoder)
decoder = autoencoder.get_layer('dec3')(decoder)
decoder = autoencoder.get_layer('dec4')(decoder)
decoder = Model(encoded_input, decoder)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(train_data, train_data, epochs=120, batch_size=5, verbose=0, validation_data=(test_data, test_data))

regres = Model(main_input, regres)
regres.compile(optimizer="adam", loss="mse")
regres.fit(train_data, train_target, epochs=80, batch_size=2, verbose=0, validation_data=(test_data, test_target))

encoded_data = encoded.predict(test_data)
decoded_data = decoder.predict(encoded_data)
predicted_data = regres.predict(test_data)

decoder.save('decoder.h5')
encoded.save('encoder.h5')
regres.save('regres.h5')

writeAsCSV("train_data.csv", np.round(train_data, 3), 1)
writeAsCSV("test_data.csv", np.round(test_data, 3), 1)
writeAsCSV("train_target.csv", np.round(train_target, 3), 0)
writeAsCSV("test_target.csv", np.round(test_target, 3), 0)
writeAsCSV("encoded_data.csv", np.round(encoded_data, 3), 1)
writeAsCSV("decoded_data.csv", np.round(decoded_data, 3), 1)
writeAsCSV("result.csv", np.round(np.column_stack((test_target, predicted_data[:, 0])), 3), 1)
