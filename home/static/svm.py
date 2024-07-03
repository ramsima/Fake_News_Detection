import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# pip install spacy
# python -m spacy download en_core_web_sm
# pip install beautifulsoup4
# pip install textblob
# pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

fake=pd.read_csv('home/static/Fake.csv')
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


real=pd.read_csv('home/static/True.csv')

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

data

X = data['text']
y= data['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train


 # TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
Xv_train = tfidf_vectorizer.fit_transform(X_train)
Xv_test = tfidf_vectorizer.transform(X_test)



# Define the SVM model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

svm = SVC()


svm.fit(Xv_train, y_train)
pred_svm = svm.predict(Xv_test)

print(pred_svm)




# Save the model
# with open('svm_model.pkl', 'wb') as file:
#     pickle.dump(svm, file)


#y_pred = best_svm.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, pred_svm)
print(accuracy)
print(classification_report(y_test, pred_svm))






