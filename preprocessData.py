
# coding: utf-8

# In[ ]:

# load in glove pretrained word vector
import numpy as np

filename = '/home/kl/projects/coattention/glove.6B/glove.6B.300d.txt'
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd
vocab,embd = loadGloVe(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
embedding = embedding.astype(np.float)


# In[ ]:




# In[ ]:

print embedding.shape


# In[ ]:

print len(vocab)


# In[ ]:

embedding.shape


# In[ ]:

print embedding_dim


# In[ ]:

# import tensorflow as tf

# # embedding layer
# W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
#                 trainable=False, name="W")
# embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
# embedding_init = W.assign(embedding_placeholder) # assign the new value to W, embeddding_init is just for sess.run


# In[ ]:

# sess = tf.Session()
# sess.run(embedding_init, feed_dict={embedding_placeholder:embedding})


# In[ ]:

# load in training data and split it into training and evaluation datasets
# load in train-v1.1.json

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import json
from random import shuffle
from spacy.en import English

nlp = English()
split_fractions = [0.8, 0.2] # just for divding the training data into training (80%) and evaluation (20%) parts
def openRawfile(input_json_file):
    #input_json_file = 'projects/coattention/train-v1.1.json'
    data_file = open(input_json_file,'r')
    parsed_file = json.load(data_file)
    data = parsed_file['data']
    len_data = len(data)
    indices = [int(len_data*(1-split)) for split in split_fractions]
    indices.insert(0,0)
    data_file.close
    return data, indices
data, indices = openRawfile('projects/coattention/train-v1.1.json')
    
   


# In[ ]:

print indices


# In[ ]:

# myDict = {'key1':1, 'key2':2, 'key3':3}
# print len(data[indices[1]:indices[2]])


# In[ ]:

# u=10
# def add(i):
#     global u
#     i = u+1
#     u-=1
#     print i
# add(0)
# add(0)
# add(0)
# print u


# In[ ]:

# list = []
# list.append([1,2])
# list.append([3,4])
# list.append([5,6])
# # print list[0:3]
# list[0].append(9)
# list = np.asarray(list)
# #print list
# #print list.shape
# list1 = [1,2,3]
# list2 = np.zeros(2)

# embeddingCut = np.zeros((1,embedding_dim)) # id index = 0 means zero padding
# print embeddingCut
# embeddingCut = np.concatenate((embeddingCut,[embedding[vocab.index('unknown')]]),axis=0) # id index = 1 means 'unknown' word

# print embeddingCut


# In[ ]:

# build vocab from context and question from training data 
# truncate the big embedding matrix to store only the word vectors for vocab words

# shuffle data by topic # by 'title'
shuffle(data)

# data parameters

c_timesteps = 600 # maximum length of sequence in context
q_timesteps = 30 # maximum length of sequence in question

#trainData = data[indices[1]:indices[2]]
trainData = data
devData = data[indices[0]:indices[1]]
#trainData = data[88:89]

from nltk.tokenize import word_tokenize

# embeddingCut = [] # to be saved and later imported, small size of the embedding matrix
embeddingCut = np.zeros((1,embedding_dim)) # id index = 0 means zero padding
embeddingCut = np.concatenate((embeddingCut,[embedding[vocab.index('unknown')]]),axis=0) # id index = 1 means 'unknown' word
myVocab = {} # temporal use for building a corresponding relationship between word in data and ids

# creat_vocab_embedding() # by calling this function to create vocab and truncated embeddding matrix
#def creat_vocab_embedding(): # can define this function for further easily preprocess data, modify later
cw = 0
wordCount = 2
# preprocess the data, extract the ids for inputs, build new myvocab, cut the embedding matrix
def preprocess(text):
    
    global embeddingCut
    global myVocab
    global wordCount
    global cw
    
    ids = []
    sentences = sent_tokenize(text)
    for sent in sentences:
        words = word_tokenize(sent)
        for word in words:
            cw+=1
            if myVocab.has_key(word):                    
                ids.append(myVocab.get(word))                    
            else:
                if word in vocab:   
                    myVocab[word] = wordCount
                    ids.append(wordCount)
                    # it is not good to have registered numpy array in the for loop, it is so time consuming 
                    embeddingCut = np.concatenate((embeddingCut,[embedding[vocab.index(word)]]),axis=0)
                    #embeddingCut.append(embedding[vocab.index(word)])
                    wordCount+=1
                else:
                    ids.append(1) # corresponding to 'unknown' word vector
    return ids

# since the answers given are char index, not the token index, 
# so change to token index by calling the function below
def toTokenIndex(context,charStart):
    context = context[0:charStart]
    sentences = sent_tokenize(context)
    count = 0
    for sent in sentences:
        words = word_tokenize(sent)
        count+= len(words)
    return count
    
context_ids = []
question_ids = []
answers = []

# pg_id is to link the question answer pairs to the corresponding context paragraph
# this id is put at the first element of each question_id and answer row
# for recognizing the context row in context_ids
pg_id = 0

# explore each word in context and question through dataset
for datum in trainData: # one datum is about one particle with one specific title
    
    for paragraph in datum['paragraphs']: # one paragraph is just one paragraph in one article, with many questions and answers
        context = nlp(paragraph['context']).text.lower()
       # context = context.encode('utf-8')
        context_id = preprocess(context)         
               
        # padding to 600 words in each context
        if len(context_id)<c_timesteps: # less than max length of a context, add padddings
            padding = [0]*(c_timesteps-len(context_id))
            context_id = context_id + padding
        else: # more than max length of a context, truncate the extra ones
            context_id = context_id[0:c_timesteps]
            
        context_ids.append(context_id)
            
        qas = paragraph['qas'] # all question-and-answer pairs for each paragraph
        for qa in qas: # each quesiton-and-answer pair
            question = nlp(qa['question']).text.lower()
            #question = question.encode('utf-8')
            question_id = preprocess(question)
            
            # padding to 30 words in each context
            if len(question_id)<q_timesteps: # less than max length of a question, add padddings
                padding = [0]*(q_timesteps-len(question_id))
                question_id = question_id + padding
            else: # more than max length of a question, truncate the extra ones
                question_id = question_id[0:q_timesteps]
                
            question_id.append(pg_id)
            question_ids.append(question_id)
            
            answer = nlp(qa['answers'][0]['text']).text.lower() # just select best one
            #answer = answer.encode('utf-8')
            answer_len = len(word_tokenize(answer))     
            start = int(qa['answers'][0]['answer_start'])
            start = toTokenIndex(context, start)
            end = start + answer_len
            answer_index = []
            answer_index.append(start)
            answer_index.append(end)
            answer_index.append(pg_id)
            answers.append(answer_index)              
            
        pg_id+=1 # pg_id is the index of the each context paragraph

# convert all the lists to be numpy array, 
# and then to be saved on disk for later importing
embeddingCut = np.asarray(embeddingCut)
context_ids = np.asarray(context_ids)
question_ids = np.asarray(question_ids)
answers = np.asarray(answers)

np.save('/home/kl/projects/data/embedingCutAllData_300d',embeddingCut)
np.save('/home/kl/projects/data/context_ids_allData_300d',context_ids)
np.save('/home/kl/projects/data/question_ids_allData_300d',question_ids)
np.save('/home/kl/projects/data/answers_allData_300d',answers)

print 'embddingCut.shape: %d', embeddingCut.shape



# In[ ]:

# np.save('projects/data/embedingCutAllData',embeddingCut)
# np.save('projects/data/context_ids_allData',context_ids)
# np.save('projects/data/question_ids_allData',question_ids)
# np.save('projects/data/answers_allData',answers)


# In[ ]:

print embeddingCut.shape
print context_ids.shape
print question_ids.shape
print answers.shape


# In[ ]:

#####################################
#    load in data from saved files
# #####################################
# import numpy as np
# # import data and truncated embedding matrix
# embeddingCutLoad = np.load('projects/coattention/embedingCutAll_correct.npy')
# all_context_ids = np.load('projects/coattention/context_ids_all_correct.npy')
# all_question_ids = np.load('projects/coattention/question_ids_all_correct.npy')
# all_answers = np.load('projects/coattention/answers_all_correct.npy')


# In[ ]:

# print embeddingCutLoad.shape
# print all_context_ids.shape
# print all_question_ids.shape
# print answers.shape


# In[ ]:

# print all_answers[100:200]


# In[ ]:



