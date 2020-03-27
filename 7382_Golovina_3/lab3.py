#Подключение модулей
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import boston_housing
import matplotlib.pyplot as plt

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#Загрузка данных
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#Нормализация
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -=mean
test_data /= std

#Настройка K-fold cross-validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 54
all_scores = []
loss_array = []
val_loss_array = []
mae_array = []
val_mae_array = []
for i in range(k):
    print('processing fold #', i)
    #выборка из данных для шага
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    #Построение модели
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    mae_array.append(history.history['mae'])
    val_mae_array.append(history.history['val_mae'])
    loss_array.append(history.history['loss'])
    val_loss_array.append(history.history['val_loss'])
    #график ошибок
    print(history.history.keys())
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    #график точности
    plt.clf()
    mae = history.history['mae']
    val_mae_graph = history.history['val_mae']
    plt.plot(epochs, mae, 'bo', label='Training mae')
    plt.plot(epochs, val_mae_graph, 'b', label='Validation mae')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()
    #Оценка
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

#средний график ошибок
epochs = range(1, num_epochs + 1)
plt.plot(epochs, np.mean(loss_array, axis=0), 'bo', label='Training loss')
plt.plot(epochs, np.mean(val_loss_array, axis=0), 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#средний график точности
plt.clf()
plt.plot(epochs, np.mean(mae_array, axis=0), 'bo', label='Training mae')
plt.plot(epochs, np.mean(val_mae_array, axis=0), 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

print(np.mean(all_scores))