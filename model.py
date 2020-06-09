from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Activation,LSTM,Dense,CuDNNLSTM, Flatten, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import os
import matplotlib.pyplot as plt
import re
np.random.seed(10)

BATCH_SIZE = 32
maxlen = 50 ##timesteps
epochs = 10
MIN_WORD_FREQUENCY = 10
song_count = 50000


df=pd.read_csv("../input/song6000/song6000.csv",engine="python")
data=np.array(df)
data[0]

text=" "
for ix in range(len(data)):
    text+=str(data[ix])
# text = text.lower()
text = text.lower().replace('\n', ' \n ')
text = re.sub(" +" , " ", text)
print('Corpus length in characters:', len(text))
corpus = [w for w in text.split(' ') if w.strip() != '' or w == '\n'
          and (w[0] not in ["(","[" ] and w[-1] not in [")","]" ])]
while "" in corpus:
    corpus.remove("")
print('Corpus length in words:', len(corpus))

text[:1000]


"""### Filtering vocabulary based on word frequency"""

word_freq = {}
for word in corpus:
    word_freq[word] = word_freq.get(word, 0) + 1

ignored_words = set()
for k, v in word_freq.items():
    if word_freq[k] < MIN_WORD_FREQUENCY:
        ignored_words.add(k)

vocab = set(corpus)
print('Unique words before ignoring:', len(vocab))
print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
vocab = sorted(set(vocab) - ignored_words)
print('Unique words after ignoring:', len(vocab))
# print_vocabulary(vocabulary, words)


"""### Creating Vocabulary and char, index mappings"""

word_ix={c:i for i,c in enumerate(vocab)}
ix_word={i:c for i,c in enumerate(vocab)}

"""### Filtering corpus based on new vocabulary"""

sentences = []
next_words = []
ignored = 0
for i in range(0, len(corpus) - maxlen):
    # Only add the sequences where no word is in ignored_words
    if len(set(corpus[i: i+maxlen+1]).intersection(ignored_words)) == 0:
        sentences.append(corpus[i: i + maxlen])
        next_words.append(corpus[i + maxlen])
    else:
        ignored = ignored + 1
print('Ignored sequences:', ignored)
print('Remaining sequences:', len(sentences))


"""### Creating the train and test datasets"""

split_count = int(0.8 * len(sentences))
sentences_test = sentences[split_count:]
next_words_test = next_words[split_count:]
sentences = sentences[:split_count]
next_words = next_words[:split_count]

"""### Check vocab size and corpus size"""

vocab_size=len(vocab) ##Dimentions of each char
print(vocab_size)

len(corpus)

def generator(sentence_list, next_word_list, batch_size):
    '''
    Generator function to generate the input/output data using
    generators concept(to avoid RAM overflow)
    '''
    index = 0
    while True:
        x = np.zeros((batch_size, maxlen, vocab_size), dtype=np.bool)
        y = np.zeros((batch_size, vocab_size), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_ix[w]] = 1
            y[i, word_ix[next_word_list[index]]] = 1

            index = index + 1
            if index == len(sentence_list):
                index = 0
        yield x, y


def create_model(timesteps, vocab_size, no_layers=3,dropout=0.2):
    '''
    Creating the model
    '''
    model=tf.keras.Sequential()
    for i in range(no_layers):
        model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True),input_shape=(timesteps,vocab_size)))
    model.add(Flatten())
    #model.add(Bidirectional(CuDNNLSTM(128), input_shape=(timesteps,vocab_size)))
    model.add(Dropout(dropout))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy')
    return model

model = create_model(maxlen, vocab_size)

keras.__version__



def sample(preds, temperature=1.0):
    '''
    helper function to sample an index from a probability array
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    '''
    Callback function to write output to file after each epoch
    '''
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(sentences+sentences_test))
    seed = (sentences+sentences_test)[seed_index]
#     print(seed)

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write("----- Generated lyrics:\n")
        examples_file.write(' '.join(sentence))

        for i in range(50):
            x_pred = np.zeros((1, maxlen, vocab_size))
#             print("sentence len: {0}".format(len(sentence)))
            for t, word in enumerate(sentence):
#                 print(word)
                x_pred[0, t,word_ix[word]] = 1

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word_pred = ix_word[next_index]

            sentence = sentence[1:]
#             print(sentence)
            sentence.append(next_word_pred)

            examples_file.write(" "+next_word_pred)
        examples_file.write('\n\n')
    examples_file.write('='*80 + '\n')
#     examples_file.flush()


"""### Opening the output file"""

examples_file = open("output_data_word.txt", "w")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


"""### Training the model"""

# Commented out IPython magic to ensure Python compatibility.
import datetime
# %load_ext tensorboard

file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % (
    len(vocab),
    maxlen,
    10
)
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)

checkpoint_path = "cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
callbacks_list = [print_callback, cp_callback]
history = model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
    steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
    epochs=epochs,
    validation_data=generator(sentences_test, next_words_test, BATCH_SIZE)
                    ,validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1,
                   callbacks = callbacks_list)
# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

"""### Closing the output file"""

examples_file.close()

"""### Plotting Train Loss curve"""

plt.plot(history.history['loss'])

"""### Plotting Validation Loss curve"""

plt.plot(history.history['val_loss'])

"""### Saving the model to disk"""

model.save('keras_model_word.hdf5')
# loaded_model = keras.models.load_model('keras_model_word.hdf5')

"""### Loading the model"""

loaded_model = tf.keras.models.load_model('keras_model_word.hdf5')



def predict_n(model, input_seq, len_out=5):
    generated = []
    actual = []
    # sent=txt[start_index:start_index+maxlen]
    sent = input_seq
    generated += sent
    gen = generated
    for i in range(len_out):
        x_sample=generated[i:i+maxlen+2]
        print(i,i+maxlen+1)
        print(x_sample)
        x = np.zeros((1,maxlen,vocab_size))
        for j in range(maxlen):
            x[0,j,word_ix[x_sample[j]]] = 1
        probs = model.predict(x)
        probs = np.reshape(probs,probs.shape[1])
        ix = np.argmax(probs)
        ix=np.random.choice(range(vocab_size),p=probs.ravel())
        generated.append(ix_word[ix])
    return " ".join(generated)



# txt = corpus
# start_index = 230
for j in range(0, 100, maxlen):
    generated = []
    actual = []
    # sent=txt[start_index:start_index+maxlen]
    sent = sentences_test[j]
    generated += sent
    actual += sent
    print("#######################")
    print("Input - "," ".join(generated))
    gen = generated
    for i in range(min(100,len(generated))):
        x_sample=generated[i:i+maxlen]
        x = np.zeros((1,maxlen,vocab_size))
        for k in range(maxlen):
            x[0,k,word_ix[x_sample[k]]] = 1
        probs = model.predict(x)
        probs = np.reshape(probs,probs.shape[1])
#         ix = np.argmax(probs)
        ix=np.random.choice(range(vocab_size),p=probs.ravel())
        generated.append(ix_word[ix])
        actual.append(next_words_test[j+i])
#         print(j)
#         print(i)
#         print(next_words_test[j+i])
#         if(i==1):
#             break
    # for i in range(100):
    #     x_sample=gen[i:i+maxlen]
    #     x=np.zeros((1,maxlen,vocab_size))
    #     for j in range(maxlen):
    #         x[0,j,char_ix[x_sample[j]]]=1
    #     probs=loaded_model.predict(x)[0]
    #     ix = np.argmax(probs)
    # #     ix=np.random.choice(range(vocab_size),p=probs.ravel())
    #     gen+=ix_char[ix]
    # # print("--------------")
    print("Actual ###############")
    print(" ".join(actual))
    print()
    print("Generated ############### ")
    print(" ".join(generated))

print("Generated ############### ")
print(" ".join(generated))
print()
print("Actual ###############")
print(" ".join(actual))


