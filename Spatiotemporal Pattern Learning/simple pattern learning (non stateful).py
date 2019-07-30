import pixiedust
from __future__ import print_function
import keras
from keras import backend as K
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

#text = ';'.join([s + '.' + s[::-1] for s in [''.join(random.choices(['A','B','C','D','E','F','G','H','I'], k=4)) for i in range(0, 100)]])
#text_test = 'ABCD.'

# text = "ABABABABABABABABABABABABABABABABABABABABABABAB"
# text_test = "BABAB"

#text = "ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB.ABB"
#text_test = "BB.ABB.ABB.A"

text = "1AABBAABB2BABABABA1AABBAABB2BABABABA1AABBAABB2BABABABA1AABBAABB2BABABABA1AABBAABB2BABABABA1AABBAABB2BABABABA"
text_test = "ABABA1AABBAAB"

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
one_hot_chars = np.eye(len(chars))

sample_length = 5 # excluding one output letter
sample_count = len(text) - sample_length - 2 # -2 instead of -1 because one output letter
char_count = len(chars)
chars_to_predict = 1
lstm_output_units = char_count * chars_to_predict

samples_x = np.zeros((sample_count, sample_length,    char_count), dtype=np.bool)
samples_y = np.zeros((sample_count, chars_to_predict, char_count), dtype=np.bool)

for sample_index in range(0, sample_count):
  sample_text = text[sample_index:sample_index + sample_length]
  print('sample_text = {}'.format(sample_text))
  for sample_char_index in range(0, sample_length):
    char = sample_text[sample_char_index]
    print('sample char = {}'.format(char))
    samples_x[sample_index, sample_char_index, char_indices[char]] = 1
  output_char = text[sample_index + sample_length + 2]
  print('output char = {}'.format(output_char))
  samples_y[sample_index, 0, char_indices[output_char]] = 1
  print('----------------------')


# print('Vectorization...')
# X_train = np.zeros((sample_count, sample_length, char_count), dtype=np.bool)
# # Y_train = np.zeros((sample_count, char_count), dtype=np.bool)
# for t, char in enumerate(text):
#     X_train[0, t, char_indices[char]] = 1
# # Y_train[0, char_indices[text[t+1]]] = 1


model = Sequential()
model.add(LSTM(lstm_output_units, batch_input_shape=(1, 1, char_count), return_sequences=False, stateful=True))
model.add(Dense(char_count, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(text)
print()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch):
    input_text = []
    output_text = []

    for i in range(0, len(text_test) - 1):
        input_char = text_test[i]
        one_hot_input = one_hot_chars[char_indices[input_char]]
        preds = model.predict(np.array([[one_hot_input]]), verbose=0)[0]

    char = text_test[-1]
    for i in range(0, 20):
        input_char = char
        one_hot_input = one_hot_chars[char_indices[input_char]]
        input_text.append(input_char)

        preds = model.predict(np.array([[one_hot_input]]), verbose=0)[0]
        print(preds)

        char = indices_char[sample(preds, 0.01)]
        output_text.append(char)

    #print('input  = {}'.format(''.join(input_text)))
    #print('output = {}'.format(''.join(output_text)))
    print('{} => {}'.format(text_test, ''.join(output_text)))


print('Train...')
for epoch in range(500):
    print('Epoch {}'.format(epoch))
    mean_tr_acc = []
    mean_tr_loss = []
    for i in range(0, len(text) - 1):
        input_char = text[i]
        output_char = text[i+1]
        #print('{} -> {}'.format(input_char, output_char))
        one_hot_input = one_hot_chars[char_indices[input_char]]
        one_hot_output = one_hot_chars[char_indices[output_char]]

        tr_loss, tr_acc = model.train_on_batch(np.array([[one_hot_input]]),
                                               np.array([one_hot_output]))
        mean_tr_acc.append(tr_acc)
        mean_tr_loss.append(tr_loss)
    model.reset_states()

    on_epoch_end(epoch)
    print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
    print('loss training = {}'.format(np.mean(mean_tr_loss)))
    print('___________________________________')



