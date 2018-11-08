
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


original_train = pd.read_csv('QuoraQuestions/train.csv')
original_test = pd.read_csv('QuoraQuestions/test.csv')


# In[3]:


original_train.head()


# In[7]:


original_test.head()


# In[9]:


len(original_train[original_train['target'] == 1])


# In[16]:


original_train[original_train['target'] == 1].head()['question_text'][30] # definitively insincere


# In[21]:


original_train[original_train['target'] == 1].head()['question_text'][110] # on the cusp - could be considered insincere bc "blacks" is not necessarily PC


# In[22]:


original_train[original_train['target'] == 1].head()['question_text'][114] # lascivious and this is borderline incest


# In[20]:


original_train[original_train['target'] == 1].head()['question_text'][115] # definitely insincere


# In[25]:


original_train[original_train['target'] == 1].tail()['question_text'][1306093] # part 2 - incest


# In[27]:


original_train[original_train['target'] == 1].tail()['question_text'][1306099] # racist towards pakistani people


# In[26]:


original_train[original_train['target'] == 1].tail()['question_text'][1306094] # provocative and trying to make a statement


# In[4]:


split_train = list(map(lambda x : x.split(), original_train[original_train['target'] == 1]['question_text']))


# In[5]:


split_train[:10]


# In[ ]:


split_train__concatenated = sum(split_train, [])
split_train_concatenated[:10]


# In[32]:


import itertools
concatenated_split_train = list(itertools.chain.from_iterable(split_train))


# In[36]:


from nltk.corpus import stopwords
s = stopwords.words('english')
concatenated_split_train = list(filter(lambda x : x not in s, concatenated_split_train))  # filter out all stop words (e.g. pronouns, articles)


# In[42]:


pd.Series(concatenated_split_train).value_counts()[:200] # top 200 words 


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
# use vectorizer to one hot encode the top 100 most common words in insincere questions among all the questions
# this will be used in training set, so that at prediction time the algorithm looks for questions with these words

