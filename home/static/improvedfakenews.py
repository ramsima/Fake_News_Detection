import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud
import tensorflow as tf


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

fake=pd.read_csv('D:\Fake_News_Detection\home\static\Fake.csv')
fake.columns
fake['subject'].value_counts()
plt.figure(figsize=(10,6))
sns.countplot(x='subject',data=fake)
text = ' '.join(fake['text'].tolist())
type(text)


wordcloud = WordCloud().generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show


real=pd.read_csv('D:\Fake_News_Detection\home\static\True.csv')
real.columns
real['subject'].value_counts()
plt.figure(figsize=(10,6))
sns.countplot(x='subject',data=real)

text = ' '.join(real['text'].tolist())


wordcloud = WordCloud().generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show



unknown_publishers =[]
for index, row in enumerate(real.text.values):
    try:
        record = row.split('-',maxsplit=1)
        record[1]
        assert(len(record[0])<120)
    except:
        unknown_publishers.append(index)



len(unknown_publishers)


real.iloc[unknown_publishers]
real = real.drop(8970, axis =0)




publisher =[]
temp_text =[]
for index, row in enumerate(real.text.values):
    if index in unknown_publishers:
        temp_text.append(row)
        publisher.append('Unknown')

    else:
        record = row.split('-', maxsplit =1)
        publisher.append(record[0].strip())
        temp_text.append(record[1].strip())


real['publisher'] = publisher
real['text'] = temp_text


real.head()


empty_fake_index = [index for index, text in enumerate(fake.text.tolist()) if str(text).strip()==""]


fake.iloc[empty_fake_index]

real['text'] = real['title'] + " " + real['text']
fake['text'] = fake['title'] + " " + fake['text']


real['text'] = real['text'].apply(lambda x: str(x).lower())
fake['text'] = fake['text'].apply(lambda x: str(x).lower())


real['class'] =1
fake['class'] = 0


real = real[['text','class']]
fake = fake[['text','class']]



data = pd.concat([real,fake], ignore_index = True)

import preprocess_kgptalkie as ps
data['text'] = data['text'].apply(lambda x: ps.remove_special_chars(x))

## For stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('English')

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def preprocess(text):
    result =[]
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token)>3 and token not in stop_words:
            result.append(token)

    return result 

data['text'] = data['text'].apply(preprocess)

data['text'] = data['text'].apply(lambda x:" ".join(x))




import gensim

y = data['class'].values


x = [d.split() for d in data['text'].tolist()]


DIM =100
w2v_model = gensim.models.Word2Vec(sentences=x, vector_size=DIM, window =10, min_count=1)


len(w2v_model.wv)


w2v_model.wv.most_similar('donald')


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)


x = tokenizer.texts_to_sequences(x)


plt.hist([len(a) for a in x], bins =700)
plt.show()


nos = np.array([len(a) for a in x])
len(nos[nos>1000])


maxlen = 1000
x = pad_sequences(x, maxlen=maxlen)


vocab_size = len(tokenizer.word_index) + 1
vocab = tokenizer.word_index


def get_weight_matrix(model):
    weight_matrix = np.zeros((vocab_size, DIM))

    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix

embedding_vectors = get_weight_matrix(w2v_model)

model = Sequential()
model.add(Embedding(vocab_size, output_dim =DIM, weights = [embedding_vectors], input_length= maxlen, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

X_train, X_test, y_train, y_test = train_test_split(x,y)

history = model.fit(X_train, y_train, validation_split=0.3,epochs=10)
# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


y_pred = (model.predict(X_test) >=0.5).astype(int)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

b = ['this is a news']
b= tokenizer.texts_to_sequences(b)
pad_sequences(x,maxlen=maxlen)

(model.predict(b)>=0.5).astype(int)


model.save('improvedfake.keras')
