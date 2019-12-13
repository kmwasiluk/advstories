from __future__ import absolute_import, division, print_function, unicode_literals

#try:
  # %tensorflow_version only exists in Colab.
  #%tensorflow_version 2.x
#except Exception:
 # pass
import tensorflow as tf

from sklearn.model_selection import train_test_split
import tracemalloc
import unicodedata
import re
import numpy as np
import os
import io
import time
import random
import sys



# Download the file
#path_to_zip = tf.keras.utils.get_file(
#    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
 #   extract=True)

path_to_file = os.path.dirname(".")+"ROCStories.txt"

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.rstrip().strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w



  # 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('<|endoftext|>')

  story_pairs = [[preprocess_sentence(w) for i in range(0,2)]  for w in lines[:num_examples]]

  return zip(*story_pairs)




def generate_random_stories(from_stories, num_examples=5000,max_length=103, min_length=50):
    ret = []
    for i in range(0,num_examples):
        app = ''
        r_int = random.randint(min_length,max_length)
        for j in range(0,r_int):
            c_story = random.choice(from_stories).split(" ")
            n_word = random.choice(c_story)
            app += ' ' + n_word
        ret.append(app)
    return ret


def max_length(tensor):
  return max(len(t) for t in tensor)





def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer


def load_dataset(path, num_examples=5000):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = create_dataset(path, num_examples)
  
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  rand_lang = generate_random_stories(targ_lang,num_examples=num_examples,max_length=max_length(input_tensor))
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
  rand_tensor, rand_lang_tokenizer = tokenize(rand_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer, rand_lang_tokenizer, rand_tensor



# Try experimenting with the size of that dataset
num_examples = 100
input_tensor, target_tensor, inp_lang, targ_lang, rand_lang, rand_tensor = load_dataset(path_to_file, num_examples)


# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
#rand_tensor = generate_random_dataset(num_examples,max_length_targ)


# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
rand_tensor_train, rand_tensor_val = train_test_split(rand_tensor,test_size=0.2)
# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))



def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))


print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[-1])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[-1])
print("random language")
convert(rand_lang, rand_tensor_train[-2])



BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 1
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 512
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
vocab_rand_size = len(rand_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train,rand_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)



example_input_batch, example_target_batch, example_rand_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape, example_rand_batch.shape




class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))




def get_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100))
    model.add(tf.keras.layers.Dropout(.3))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dropout(.3))
    model.add(tf.keras.layers.Dense(1))
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output),real_output) + cross_entropy(tf.zeros_like(fake_output),fake_output)

def fake_encoder_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

discriminator = get_discriminator()

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)



encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

fake_encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
fake_encoder_optimizer = tf.keras.optimizers.Adam(1e-4)
fake_sample_hidden = fake_encoder.initialize_hidden_state()
fake_sample_output, fake_sample_hidden = fake_encoder(example_rand_batch, fake_sample_hidden)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
print('fake sample shape: () {}'.format(fake_sample_hidden.shape))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


real_encoder_optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(real_encoder_optimizer=real_encoder_optimizer,
                                 fake_encoder_optimizer=fake_encoder_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 real_encoder=encoder,
                                 discriminator=discriminator,
                                 fake_encoder=fake_encoder,
                                 decoder=decoder)


@tf.function
def train_step(inp, targ, rand, enc_hidden):
  loss = 0
  fake_enc_loss = 0
  discr_loss = 0
  with tf.GradientTape() as real_enc_tape, tf.GradientTape() as fake_enc_tape, tf.GradientTape() as discr_tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    
    #our addition
    fake_enc_output, fake_enc_hidden = fake_encoder(rand, enc_hidden)
    real_output = discriminator(enc_hidden, training=True)
    fake_output = discriminator(fake_enc_hidden, training=True)
    discr_loss = discriminator_loss(real_output,fake_output)
    fake_enc_loss = fake_encoder_loss(fake_output)
   # real_encoder_additional_loss = fake_encoder_loss(real_output) #fel is same.
   # loss += real_encoder_additional_loss
    
    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  real_encoder_variables = encoder.trainable_variables + decoder.trainable_variables
  fake_encoder_variables = fake_encoder.trainable_variables
  discriminator_variables = discriminator.trainable_variables
  print(sys.getsizeof(real_encoder_variables))
  print(sys.getsizeof(fake_encoder_variables))
  print(sys.getsizeof(discriminator_variables))
  real_encoder_gradients = real_enc_tape.gradient(loss, real_encoder_variables)
  fake_encoder_gradients = fake_enc_tape.gradient(fake_enc_loss, fake_encoder_variables)
  discriminator_gradients = discr_tape.gradient(discr_loss, discriminator_variables)

  real_encoder_optimizer.apply_gradients(zip(real_encoder_gradients, real_encoder_variables))
  fake_encoder_optimizer.apply_gradients(zip(fake_encoder_gradients, fake_encoder_variables))
  discriminator_optimizer.apply_gradients(zip(dscriminator_gradients, discriminator_variables))


  return batch_loss


EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp,targ,rand)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp,targ,rand,enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


