import io
from django.shortcuts import render,redirect
from django.http import HttpResponse
import tensorflow as tf
from contextlib import redirect_stdout
from bs4 import BeautifulSoup
import requests


from sklearn.metrics.pairwise import cosine_similarity







def scrape_news(news_headline):
  url = 'https://news.google.com/search?q='+ news_headline + '&hl=en-US&gl=US&ceid=US%3Aen'

  page = requests.get(url)

  articles = []
  article_link=[]
  # Check if the request was successful
  if page.status_code == 200:
      # Parse the HTML content with BeautifulSoup
      soup = BeautifulSoup(page.text, 'lxml')
      
      # Find all the news titles on the page
      for item in soup.find_all('a',class_='JtKRv',href=True):
          title = item.get_text()
          href = 'https://news.google.com'+item['href']
          article_link.append(href)
          articles.append(title)
          
  else:
      print("Failed to retrieve the page. Status code:", page.status_code)
      
  return articles,article_link


# Create your views here.

def home(request):
  return render(request, "index.html")


def test(request):
  if request.method == 'POST':
    fake_news_model = request.POST.get('model')
    news = request.POST.get('news')
    if fake_news_model == "Real-Time Checking":
      
      from transformers import BertTokenizer, BertModel
      import torch
      import openai
      from sentence_transformers import SentenceTransformer, util
      
      articles,article_link = scrape_news(news)
      
      zipped_list=zip(articles,article_link)
      
      if(len(articles) != 0 ):
        #################################-----SBERT------#################################
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        model1 = SentenceTransformer('all-MiniLM-L6-v2')
        
        
        # Function to get SBERT embedding
        def get_sbert_embedding(text):
            return model1.encode(text, convert_to_tensor=True)

        # Get embedding for the user input (news)
        user_embedding1 = get_sbert_embedding(news)

        # Get embeddings for all scraped articles
        scraped_embeddings1 = [get_sbert_embedding(article) for article in articles]

        # Calculate cosine similarity for each scraped news article
        similarity_scores1 = [(index, util.cos_sim(user_embedding1, embedding).item()) 
                            for index, embedding in enumerate(scraped_embeddings1)]

        # Sort the scores in descending order
        sorted_scores = sorted(similarity_scores1, key=lambda x: x[1], reverse=True)
        
        # Output the sorted scores
        print(sorted_scores)
        
        #################################-----BERT------#################################
        
        # def get_bert_embedding(text):
        #   inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        #   with torch.no_grad():
        #       outputs = model(**inputs)
        #   return outputs.last_hidden_state.mean(dim=1).numpy()
        
        # user_embedding = get_bert_embedding(news)
        # scraped_embeddings = [get_bert_embedding(article) for article in articles]
        
        
        # # Calculate cosine similarity for each scraped news article
        # similarity_scores = [(index, cosine_similarity(user_embedding, embedding).flatten()[0]) 
        #                     for index, embedding in enumerate(scraped_embeddings)]

        # # Sort the scores in descending order
        # sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # print(sorted_scores)
        
        #################################-----GPT------#################################
        
        # Set your OpenAI API key
        # openai.api_key = 'your-api-key'

        # # Function to get GPT embedding
        # def get_gpt_embedding(text):
        #     response = openai.Embedding.create(
        #         input=text,
        #         model="text-embedding-ada-002"  # or "text-similarity-davinci-001"
        #     )
        #     return np.array(response['data'][0]['embedding'])
          
          
        #  # Get embedding for the user input (news)
        # user_embedding = get_gpt_embedding(news)

        # # Get embeddings for all scraped articles
        # scraped_embeddings = [get_gpt_embedding(article) for article in articles]

        # # Calculate cosine similarity for each scraped news article
        # similarity_scores = [(index, cosine_similarity(user_embedding.reshape(1, -1), embedding.reshape(1, -1)).flatten()[0]) 
        #                     for index, embedding in enumerate(scraped_embeddings)]

        # # Sort the scores in descending order
        # sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # # Output the sorted scores
        # print(sorted_scores)
        
        if(len(sorted_scores) >= 10):
          avg_similarity_score_top_10 = 0
          
          sorted_top_10_scores = [sorted_scores[0],sorted_scores[1],sorted_scores[2]]
          
          
          for index, score in sorted_top_10_scores:
            print(score)
            avg_similarity_score_top_10 = avg_similarity_score_top_10 + score
        
          avg_similarity_score_top_10 = avg_similarity_score_top_10/3
        else:
          avg_similarity_score_top_10 = 0
          
          sorted_top_10_scores = []
          for i in range(len(sorted_scores)):
            sorted_top_10_scores.append(sorted_scores[i])
            
          
          for index, score in sorted_top_10_scores:
            print(score)
            avg_similarity_score_top_10 = avg_similarity_score_top_10 + score

          avg_similarity_score_top_10 = avg_similarity_score_top_10/len(sorted_scores)
        
        
          
     
        
        
        
        
        print(avg_similarity_score_top_10)
        
        context = {'news':news,
                   'result':avg_similarity_score_top_10,
                   'model_selected':fake_news_model,
                   'articles':articles,
                   'zipped_list':zipped_list
                   }

        # # Get the most similar article
        # most_similar_index, highest_score = sorted_scores[0]
        

        # # Output the most similar article
        # print(f"Most Similar Article {most_similar_index + 1}:")
        # print(f"Similarity Score = {highest_score}")
        # print(f"Text: {articles[most_similar_index]}\n")

        

        # Output the most similar articles in descending order of similarity
        # for index, score in sorted_scores:
        #     print(f"Article {index + 1}: Similarity Score = {score}")
        #     print(f"Text: {articles[index]}\n")

      else:
        result1 = "Sorry, There is not a single related news articles!"
        context = {'news':news,
                   'result':result1,
                   'model_selected':fake_news_model}
      
      return render(request, 'realtimechecking.html', context)
      
      
      
    if fake_news_model == "LSTM":
      import tensorflow as tf
      from sklearn.feature_extraction.text import TfidfVectorizer
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
      
      # print(lstm_model.summary())
      
      # Load the tokenizer from the file
      with open('home/static/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

      
      # ranked_indices = similarities.argsort()[::-1]
      # print(ranked_indices)

      # Display the ranked articles and their similarity scores
      # for index in ranked_indices:
      #     print(f"Article: {articles[index]}")
      #     print(f"Similarity: {similarities[index]:.4f}\n")
      
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
      
      return render(request, 'othermodelchecking.html', context)
    
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
      
      
      
      
      return render(request,'othermodelchecking.html', context)
    
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
      nltk.download('stopwords')
      stop_words = ['i',
                    'me',
                    'my',
                    'myself',
                    'we',
                    'our',
                    'ours',
                    'ourselves',
                    'you',
                    "you're",
                    "you've",
                    "you'll",
                    "you'd",
                    'your',
                    'yours',
                    'yourself',
                    'yourselves',
                    'he',
                    'him',
                    'his',
                    'himself',
                    'she',
                    "she's",
                    'her',
                    'hers',
                    'herself',
                    'it',
                    "it's",
                    'its',
                    'itself',
                    'they',
                    'them',
                    'their',
                    'theirs',
                    'themselves',
                    'what',
                    'which',
                    'who',
                    'whom',
                    'this',
                    'that',
                    "that'll",
                    'these',
                    'those',
                    'am',
                    'is',
                    'are',
                    'was',
                    'were',
                    'be',
                    'been',
                    'being',
                    'have',
                    'has',
                    'had',
                    'having',
                    'do',
                    'does',
                    'did',
                    'doing',
                    'a',
                    'an',
                    'the',
                    'and',
                    'but',
                    'if',
                    'or',
                    'because',
                    'as',
                    'until',
                    'while',
                    'of',
                    'at',
                    'by',
                    'for',
                    'with',
                    'about',
                    'against',
                    'between',
                    'into',
                    'through',
                    'during',
                    'before',
                    'after',
                    'above',
                    'below',
                    'to',
                    'from',
                    'up',
                    'down',
                    'in',
                    'out',
                    'on',
                    'off',
                    'over',
                    'under',
                    'again',
                    'further',
                    'then',
                    'once',
                    'here',
                    'there',
                    'when',
                    'where',
                    'why',
                    'how',
                    'all',
                    'any',
                    'both',
                    'each',
                    'few',
                    'more',
                    'most',
                    'other',
                    'some',
                    'such',
                    'no',
                    'nor',
                    'not',
                    'only',
                    'own',
                    'same',
                    'so',
                    'than',
                    'too',
                    'very',
                    's',
                    't',
                    'can',
                    'will',
                    'just',
                    'don',
                    "don't",
                    'should',
                    "should've",
                    'now',
                    'd',
                    'll',
                    'm',
                    'o',
                    're',
                    've',
                    'y',
                    'ain',
                    'aren',
                    "aren't",
                    'couldn',
                    "couldn't",
                    'didn',
                    "didn't",
                    'doesn',
                    "doesn't",
                    'hadn',
                    "hadn't",
                    'hasn',
                    "hasn't",
                    'haven',
                    "haven't",
                    'isn',
                    "isn't",
                    'ma',
                    'mightn',
                    "mightn't",
                    'mustn',
                    "mustn't",
                    'needn',
                    "needn't",
                    'shan',
                    "shan't",
                    'shouldn',
                    "shouldn't",
                    'wasn',
                    "wasn't",
                    'weren',
                    "weren't",
                    'won',
                    "won't",
                    'wouldn',
                    "wouldn't"]
      
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
      
      with open('home/static/svm_tfidf_vectorizer.pkl', 'rb') as f:
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
      
      return render(request, "othermodelchecking.html", context)
      
      
    
    elif fake_news_model == "Logistic Regression":
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
      nltk.download('stopwords')
      stop_words = ['i',
                    'me',
                    'my',
                    'myself',
                    'we',
                    'our',
                    'ours',
                    'ourselves',
                    'you',
                    "you're",
                    "you've",
                    "you'll",
                    "you'd",
                    'your',
                    'yours',
                    'yourself',
                    'yourselves',
                    'he',
                    'him',
                    'his',
                    'himself',
                    'she',
                    "she's",
                    'her',
                    'hers',
                    'herself',
                    'it',
                    "it's",
                    'its',
                    'itself',
                    'they',
                    'them',
                    'their',
                    'theirs',
                    'themselves',
                    'what',
                    'which',
                    'who',
                    'whom',
                    'this',
                    'that',
                    "that'll",
                    'these',
                    'those',
                    'am',
                    'is',
                    'are',
                    'was',
                    'were',
                    'be',
                    'been',
                    'being',
                    'have',
                    'has',
                    'had',
                    'having',
                    'do',
                    'does',
                    'did',
                    'doing',
                    'a',
                    'an',
                    'the',
                    'and',
                    'but',
                    'if',
                    'or',
                    'because',
                    'as',
                    'until',
                    'while',
                    'of',
                    'at',
                    'by',
                    'for',
                    'with',
                    'about',
                    'against',
                    'between',
                    'into',
                    'through',
                    'during',
                    'before',
                    'after',
                    'above',
                    'below',
                    'to',
                    'from',
                    'up',
                    'down',
                    'in',
                    'out',
                    'on',
                    'off',
                    'over',
                    'under',
                    'again',
                    'further',
                    'then',
                    'once',
                    'here',
                    'there',
                    'when',
                    'where',
                    'why',
                    'how',
                    'all',
                    'any',
                    'both',
                    'each',
                    'few',
                    'more',
                    'most',
                    'other',
                    'some',
                    'such',
                    'no',
                    'nor',
                    'not',
                    'only',
                    'own',
                    'same',
                    'so',
                    'than',
                    'too',
                    'very',
                    's',
                    't',
                    'can',
                    'will',
                    'just',
                    'don',
                    "don't",
                    'should',
                    "should've",
                    'now',
                    'd',
                    'll',
                    'm',
                    'o',
                    're',
                    've',
                    'y',
                    'ain',
                    'aren',
                    "aren't",
                    'couldn',
                    "couldn't",
                    'didn',
                    "didn't",
                    'doesn',
                    "doesn't",
                    'hadn',
                    "hadn't",
                    'hasn',
                    "hasn't",
                    'haven',
                    "haven't",
                    'isn',
                    "isn't",
                    'ma',
                    'mightn',
                    "mightn't",
                    'mustn',
                    "mustn't",
                    'needn',
                    "needn't",
                    'shan',
                    "shan't",
                    'shouldn',
                    "shouldn't",
                    'wasn',
                    "wasn't",
                    'weren',
                    "weren't",
                    'won',
                    "won't",
                    'wouldn',
                    "wouldn't"]
      
      import gensim
      from gensim.utils import simple_preprocess
      from gensim.parsing.preprocessing import STOPWORDS

      def preprocess(text):
          result1 =[]
          for token in gensim.utils.simple_preprocess(text):
              if token not in gensim.parsing.preprocessing.STOPWORDS and len(token)>3 and token not in stop_words:
                  result1.append(token)

          return result1 
        
      with open('home/static/LR_model.pkl', 'rb') as file:
        # Use pickle.load to read the data from the file
        LR_model = pickle.load(file)
      
      with open('home/static/LR_tfidf_vectorizer.pkl', 'rb') as f:
        LR_tfidf_vectorizer = pickle.load(f)
        
        
      news_processed = news.lower()  # Convert to lowercase
      news_processed = ps.remove_special_chars(news_processed)  # Assuming this is a function for special character removal
      news_processed = preprocess(news_processed)
      news_processed_sentence = " ".join(news_processed)
      print(news_processed_sentence)
         
      # Converting words into Vectors
      news_vector = LR_tfidf_vectorizer.transform([news_processed_sentence])
      
      result =  LR_model.predict(news_vector)
      
      print(news)
      print(result)
      
      context = {'news':news,
                'result':result,
                'model_selected':fake_news_model}
      
      return render(request, "othermodelchecking.html", context)
    
      
    
  else:    
    return render(request, "othermodelchecking.html" )
    




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
  
  
  
def compare(request):

  
  return render(request,'compare.html')
    

