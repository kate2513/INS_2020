import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras import models
from keras import layers
from keras.datasets import imdb
import string

#ф-я для преобразования всех обзоров в векторы одного размера
def vectorize(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def str_to_list(review, dictionary):
    tt = str.maketrans(dict.fromkeys(string.punctuation))
    review = review.translate(tt)
    temp_list = review.lower().split()
    for i in range(len(temp_list)):
        if temp_list[i] in dictionary:
            temp_list[i] = dictionary[temp_list[i]]
        else:
            temp_list[i] = 0
    return temp_list

text_length = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=text_length)
#объединение тестовых и тренировочных данных
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

#length = [len(i) for i in data]
#расшифровка обзора
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
#print(index)
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )
#print(decoded)

#преобразование данных
data = vectorize(data, text_length)
targets = np.array(targets).astype("float32")
print(type(data))

#разделение датасета в отношении 80/20
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

print(type(test_x))

#построение модели
model = Sequential()
# Input - Layer
model.add(layers.Dense(50, activation = "relu", input_shape=(text_length, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
#model.summary()

#настройка параметров
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

#обучение модели
history = model.fit(
 train_x, train_y,
 epochs= 2,
 batch_size = 100,
 validation_data = (test_x, test_y)
)

#оценка
print(np.mean(history.history['val_accuracy']))

#свои строчки
reviews = [
    'Movie is great. It was very exciting.',
    'Bad movie. I hate it.',
    'It is an ordinary film. I saw tons of it. Nothing special.',
    'I love this film. It is my favourite now!'
    ]
for i in range(len(reviews)):
    reviews[i] = str_to_list(reviews[i], index)

reviews = vectorize(reviews, text_length)
prediction = model.predict(reviews)
print(prediction)
for i in range(len(prediction)):
    if prediction[i]>=0.5:
        print(i+1,": positive review")
    else:
        print(i+1,": negative review")

#график ошибок
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#график точности
plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()