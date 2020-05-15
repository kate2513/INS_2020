import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import string

def create_model(num, top_words, max_review_length):
    embedding_vecor_length = 32
    model = Sequential()
    if num == 1: 
        model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(LSTM(100))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
    if num == 2:
        model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(LSTM(100))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))    
    return model

def graphics(history):
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

def get_acc(test_y, predictions):
    all = len(test_y)
    correct = 0
    for i in range(len(test_y)):
        if test_y[i] == predictions[i]:
            correct = correct+1
    return correct/all

def test_custom_reviews(reviews, dictionary, max_review_length):
    for i in range(len(reviews)):
        reviews[i] = str_to_list(reviews[i], dictionary)
    reviews = sequence.pad_sequences(reviews, maxlen=max_review_length)
    num_models = 2
    models = load_models(num_models)
    separate_predictions = [model.predict(reviews) for model in models]
    predictions = np.array(separate_predictions).mean(axis = 0)
    for i in range(num_models):
        print("model",i+1," prediction: ",separate_predictions[i])
    print("ensemle prediction: ", predictions)
    
    
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

def load_models(num):
    models = list()
    for i in range(1,num+1):
        model = load_model('model'+str(i)+'.h5')
        models.append(model)
    return models

#data
from keras.datasets import imdb
top_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

#разделение датасета в отношении 80/20
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

#дополнить каждый отзыв до размера max_review_length
max_review_length = 500
train_x = sequence.pad_sequences(train_x, maxlen=max_review_length)
test_x = sequence.pad_sequences(test_x, maxlen=max_review_length)

#network
#first model = 1, second model = 2
models = list()
for i in range(1,3):
    try:
        model = load_model('model'+str(i)+'.h5')
    except OSError:
        model = create_model(i,top_words,max_review_length)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print(model.summary())
        history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
        epochs=2,batch_size=64)
        graphics(history)
        model.save("model"+str(i)+".h5")
    #оценка
    test_loss, test_acc = model.evaluate(test_x, test_y)
    print('model',i+1,' test_acc:', test_acc,'\ntest_loss:', test_loss)
    models.append(model)


# make predictions
predictions = [model.predict(test_x) for model in models]
predictions = np.array(predictions).mean(axis = 0)
predictions = np.round(predictions).astype(int)
predictions = predictions.flatten()
print("Ensemble acc: ", get_acc(test_y, predictions))

#custom reviews
reviews = [
    'Movie is great. It was very exciting.',
    'Bad movie. I hate it.',
    'It is an ordinary film. I saw tons of it. Nothing special.',
    'I love this film. It is my favourite now!'
    ]
index = imdb.get_word_index()    
test_custom_reviews(reviews, index, max_review_length)