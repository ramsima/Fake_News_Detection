import io
from django.shortcuts import render,redirect
from django.http import HttpResponse
import tensorflow as tf
from contextlib import redirect_stdout


# Create your views here.

def home(request):
  return render(request, "index.html")


def test(request):
  if request.method == 'POST':
    fake_news_model = request.POST.get('model')
    news = request.POST.get('news')
    if fake_news_model == "LSTM":
      import tensorflow as tf
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      import seaborn as sns
      import nltk
      import re
      from wordcloud import WordCloud
      import tensorflow as tf
      from gensim.models import Word2Vec
      import gensim
      from tensorflow.keras.preprocessing.text import Tokenizer
      from tensorflow.keras.preprocessing.sequence import pad_sequences
      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import classification_report, accuracy_score
      import pickle

      # Load the model
      lstm_model = tf.keras.models.load_model('home/static/imporvedfake.keras')

      summary_str = io.StringIO()
      with redirect_stdout(summary_str):
        lstm_model.summary()
      
      # Load the tokenizer from the file
      with open('home/static/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
      news = news.lower()
      news_list = [news]
      maxlen = 1000
      news_list= tokenizer.texts_to_sequences(news_list)
      news_list = pad_sequences(news_list,maxlen=maxlen)
      result = (lstm_model.predict(news_list)>=0.5).astype(int)
      print(result)
      print(fake_news_model)
      
      context = {'news':news,
                'result':result,
                'model_selected':fake_news_model}
      
      return render(request, 'test.html', context)
    
    elif fake_news_model == "BERT":
      import numpy as np
      import pandas as pd
      import pycaret
      import transformers
      from transformers import AutoModel, BertTokenizerFast
      import matplotlib.pyplot as plt
      from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import classification_report
      import torch
      import torch.nn as nn
      
      # Load BERT model and tokenizer via HuggingFace Transformers
      bert = AutoModel.from_pretrained('bert-base-uncased')
      tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
      
      class BERT_Arch(nn.Module):
        def __init__(self, bert):
          super(BERT_Arch, self).__init__()
          self.bert = bert
          self.dropout = nn.Dropout(0.1)            # dropout layer
          self.relu =  nn.ReLU()                    # relu activation function
          self.fc1 = nn.Linear(768,512)             # dense layer 1
          self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
          self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
        def forward(self, sent_id, mask):           # define the forward pass
          cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                    # pass the inputs to the model
          x = self.fc1(cls_hs)
          x = self.relu(x)
          x = self.dropout(x)
          x = self.fc2(x)                           # output layer
          x = self.softmax(x)                       # apply softmax activation
          return x

      model = BERT_Arch(bert)
          
      
      # load weights of best model
      path = 'home/static/BERT_model_weights.pt'
      model.load_state_dict(torch.load(path))
      
      # testing on user data
      news = request.POST.get('news')
      print(news)
      print(fake_news_model)
      
      user_news_text = [news]

      # tokenize and encode sequences in the test set
      MAX_LENGHT = 500
      tokens_unseen = tokenizer.batch_encode_plus(
          user_news_text,
          max_length = MAX_LENGHT,
          pad_to_max_length=True,
          truncation=True
      )

      user_news_text_seq = torch.tensor(tokens_unseen['input_ids'])
      user_news_text_mask = torch.tensor(tokens_unseen['attention_mask'])

      with torch.no_grad():
        preds = model(user_news_text_seq, user_news_text_mask)
        preds = preds.detach().cpu().numpy()

      preds = np.argmax(preds, axis = 1)
      if preds == 0:
        preds1 = 1
      else:
        preds1 = 0
      
      print(preds1)
      
      context = {'news':news,
                'result':preds1,
                'model_selected':fake_news_model}      
      
      
      
      
      return render(request,'test.html', context)
    
    elif fake_news_model == "SVM":
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
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import classification_report, accuracy_score
      import preprocess_kgptalkie as ps
      from nltk.corpus import stopwords
      stop_words = stopwords.words('English')
      
      import gensim
      from gensim.utils import simple_preprocess
      from gensim.parsing.preprocessing import STOPWORDS

      def preprocess(text):
          result1 =[]
          for token in gensim.utils.simple_preprocess(text):
              if token not in gensim.parsing.preprocessing.STOPWORDS and len(token)>3 and token not in stop_words:
                  result1.append(token)

          return result1 
        
      with open('home/static/svm_model.pkl', 'rb') as file:
        # Use pickle.load to read the data from the file
        svm_model = pickle.load(file)
      
      with open('home/static/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        
        
      news_processed = news.lower()  # Convert to lowercase
      news_processed = ps.remove_special_chars(news_processed)  # Assuming this is a function for special character removal
      news_processed = preprocess(news_processed)
      news_processed_sentence = " ".join(news_processed)
      print(news_processed_sentence)
         
      # Converting words into Vectors
      news_vector = tfidf_vectorizer.transform([news_processed_sentence])
      
      result =  svm_model.predict(news_vector)
      
      print(news)
      print(result)
      
      context = {'news':news,
                'result':result,
                'model_selected':fake_news_model}
      
      return render(request, "test.html", context)
      
      
      
      
      
      
    
    elif fake_news_model == "Logistic Regression":
      pass
      
    
  else:    
    return render(request, "test.html" )
    




# def testbert(request):
#   if request.method == 'POST':
#     import numpy as np
#     import pandas as pd
#     import pycaret
#     import transformers
#     from transformers import AutoModel, BertTokenizerFast
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import classification_report
#     import torch
#     import torch.nn as nn
    
#     # Load BERT model and tokenizer via HuggingFace Transformers
#     bert = AutoModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
#     class BERT_Arch(nn.Module):
#       def __init__(self, bert):
#         super(BERT_Arch, self).__init__()
#         self.bert = bert
#         self.dropout = nn.Dropout(0.1)            # dropout layer
#         self.relu =  nn.ReLU()                    # relu activation function
#         self.fc1 = nn.Linear(768,512)             # dense layer 1
#         self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
#         self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
#       def forward(self, sent_id, mask):           # define the forward pass
#         cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
#                                                   # pass the inputs to the model
#         x = self.fc1(cls_hs)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)                           # output layer
#         x = self.softmax(x)                       # apply softmax activation
#         return x

#     model = BERT_Arch(bert)
        
    
#     # load weights of best model
#     path = 'home/static/BERT_model_weights.pt'
#     model.load_state_dict(torch.load(path))
    
#     # testing on user data
#     news = request.POST.get('news')
#     print(news)
    
#     user_news_text = [news]

#     # tokenize and encode sequences in the test set
#     MAX_LENGHT = 500
#     tokens_unseen = tokenizer.batch_encode_plus(
#         user_news_text,
#         max_length = MAX_LENGHT,
#         pad_to_max_length=True,
#         truncation=True
#     )

#     user_news_text_seq = torch.tensor(tokens_unseen['input_ids'])
#     user_news_text_mask = torch.tensor(tokens_unseen['attention_mask'])

#     with torch.no_grad():
#       preds = model(user_news_text_seq, user_news_text_mask)
#       preds = preds.detach().cpu().numpy()

#     preds = np.argmax(preds, axis = 1)
#     print(preds)
    
#     context = {'result':preds}
    
    
#     return render(request,'test2.html', context)
#   else:
#     return render(request,'test2.html')
  
  
  
def about(request):
  return render(request,'about.html')
    

