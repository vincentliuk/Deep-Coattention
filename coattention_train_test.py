
# coding: utf-8

# In[1]:


#####################################
#    load in data from saved files
#####################################

import os
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# import data and truncated embedding matrix
embeddingCutLoad = np.load('/home/liuk/coattention/data/embedingCutTrainData_300d.npy')
all_context_ids = np.load('/home/liuk/coattention/data/context_ids_trainData_300d.npy')
all_question_ids = np.load('/home/liuk/coattention/data/question_ids_trainData_300d.npy')
all_answers = np.load('/home/liuk/coattention/data/answers_trainData_300d.npy')

dev_context_ids = np.load('/home/liuk/coattention/data/context_ids_testData_300d.npy')
dev_question_ids = np.load('/home/liuk/coattention/data/question_ids_testData_300d.npy')

dev_size = dev_question_ids.shape[0]

# batch_linear func, for a nonlinear pass of question embedding to generate a variation

def batch_linear(inputs, output_size, bias=True, scope=None):
   
    shape = inputs.get_shape() # shape (1, 31, 200) or (1, 1200,1)
    m = shape[1].value # m = 31 or 1200
    n = shape[2].value # n = 200 or 1
    dtype = inputs.dtype # float32

    weights = tf.get_variable(
        "bl_weights", [output_size, n], 
        dtype=dtype,
        initializer = tf.random_normal_initializer()) # shape (31, 31)
    res = tf.map_fn(lambda x: tf.matmul(x, weights), inputs)
    if not bias: # no bias, just return the res
        return res

    biases = tf.get_variable(
        "bl_biases", [m, n],
        dtype=dtype,
        initializer=tf.constant_initializer(0.0))
    return tf.map_fn(lambda x: tf.add(x, biases), res) # element-wise


##########################
#    Model Encoding 
##########################

vocab_size = embeddingCutLoad.shape[0]
embedding_dim = embeddingCutLoad.shape[1]

# most used variables
batch_size = 50 # batch_size = 1 -- kl
hidden_size = 200 # 200
min_timesteps = 30 #30
max_timesteps = 600 #600

batch_context_ids_data = np.zeros((batch_size,max_timesteps),dtype=np.int)
batch_question_ids_data = np.zeros((batch_size,min_timesteps),dtype = np.int)
batch_answer_data = np.zeros((batch_size,2),dtype=np.int)


# tf graph inputs, creating placeholders
# set the batch_size to be None, which can help to receive batches of data with different size
batch_context_ids = tf.placeholder(tf.int32,[None,max_timesteps])
batch_question_ids = tf.placeholder(tf.int32,[None, min_timesteps])
batch_answer = tf.placeholder(tf.int64,[None,2])
batch_guesses = tf.placeholder(tf.int64, [None,2])

embeddingMatrix = tf.placeholder(tf.float32, [vocab_size, embedding_dim])

batch_guesses_initial = np.zeros((batch_size,2),dtype=np.int)


def build_encoder(context_ids,question_ids):
    """Builds coattention encoder."""
    
    # global U
    
    with tf.variable_scope('embedding'):
        # fixed embedding      
        # embed c_inputs and q_inputs.
        fn = lambda x: tf.nn.embedding_lookup(embeddingMatrix, x) # define a function with arg: x to look up embedding table -- kl
        c_embedding = tf.map_fn(lambda x: fn(x), context_ids, dtype=tf.float32) #shape (batch_size,600,50) real data coming here - kl
        q_embedding = tf.map_fn(lambda x: fn(x), question_ids, dtype=tf.float32) # real data coming here
        
        #scope.reuse_variables()

        # shared lstm encoder
        lstm_enc = tf.contrib.rnn.LSTMCell(hidden_size) # has not input data in yet -- kl
        ### done above ###
    
    with tf.variable_scope('c_embedding'):
    # compute context embedding # by using the lstm_enc created in line 123, lstm_enc is a LSTM cell
        # mask can be added here in the dynamic_rnn
        c, _ = tf.nn.dynamic_rnn(lstm_enc, c_embedding, dtype=tf.float32) # c outputs - kl
        # append sentinel # paper page 2 2.1 first paragraph
        fn = lambda x: tf.concat(
            [x, tf.zeros([1, hidden_size], dtype=tf.float32)], 0)
        c_encoding = tf.map_fn(lambda x: fn(x), c, dtype=tf.float32) # shape (1, 601, 200)
        
    with tf.variable_scope('q_embedding'):
        # compute question embedding
         # mask can be added here in dynamic_rnn
        q, _ = tf.nn.dynamic_rnn(lstm_enc, q_embedding, dtype=tf.float32)
        # append sentinel
        fn = lambda x: tf.concat(
            [x, tf.zeros([1, hidden_size], dtype=tf.float32)],0)
        q_encoding = tf.map_fn(lambda x: fn(x), q, dtype=tf.float32) # shape = (1,31,200)
        # allow variation between c_embedding and q_embedding
        with tf.variable_scope('non_linearity'):
            q_encoding = tf.tanh(batch_linear(q_encoding, hidden_size, True)) # shape (1, 31, 200)
            # q_variation is for calculating the coattention matrix, make q transpose firstly
            q_variation = tf.transpose(q_encoding, perm=[0, 2, 1]) # shape (1, 200, 31)
    
    with tf.variable_scope('coattention'):
        # compute affinity matrix, (batch_size, context+1, question+1)
          # c_encoding: [batch_size,c_timesteps,hidden_dimension]
          # q_variation: [batch_size, hidden_dimension, q_timesteps] 
        L = tf.matmul(c_encoding, q_variation) # shape (1, 601, 31), [batch_size, c_timesteps, q_timesteps]
        # shape = (batch_size, question+1, context+1)
        L_t = tf.transpose(L, perm=[0, 2, 1]) # shape (1, 31, 601) equation (1) L and L's transpose
        # normalize with respect to question
        a_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32) # shape (1, 31, 601)
        # normalize with respect to context
        a_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)# shape (1, 601, 31)
        # summaries with respect to question, (batch_size, question+1, hidden_size)
        c_q = tf.matmul(a_q, c_encoding)# shape (1, 31, 200)
        c_q_emb = tf.concat([q_variation, tf.transpose(c_q, perm=[0, 2 ,1])],1) # shape (1, 400, 31)
        # summaries of previous attention with respect to context
        c_d = tf.matmul(c_q_emb, a_c, adjoint_b=True,)# shape (1, 400, 601)
        # final coattention context, (batch_size, context+1, 3*hidden_size)
        co_att = tf.concat([c_encoding, tf.transpose(c_d, perm=[0, 2, 1])],2)# shape (1, 601, 600)
    
    with tf.variable_scope('encoder'):
        # LSTM for coattention encoding # bi-directional LSTM in figure 2 generating U: ut -kl
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size) # fw: forwards
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size) # bw: backwards
        # compute coattention encoding
        u, _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw, cell_bw, co_att,
          sequence_length=tf.to_int64([max_timesteps]*batch_size),
          dtype=tf.float32)
        U = tf.concat(u, 2) # shape (1, 601, 400) the final U generated in figure 2 - kl
                            # so U's shape=[batch_size, context_number+1, double hidden dimension]
    
    
    return U


#####################
##  Model Decoding ##
#####################

max_decode_steps = 4

n_input = 1600
n_hidden_1 = hidden_size
n_hidden_2 = hidden_size
n_classes = 1

### two layer MLP ###
# two Layer MLP parameters initialization
weightsAlpha = {
'Alpha_MLP_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
'Alpha_MLP_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
'Alpha_MLP_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biasesAlpha = {
'Alpha_MLP_b1': tf.Variable(tf.random_normal([n_hidden_1])),
'Alpha_MLP_b2': tf.Variable(tf.random_normal([n_hidden_2])),
'Alpha_MLP_out': tf.Variable(tf.random_normal([n_classes]))
}  
weightsBeta = {
'Beta_MLP_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
'Beta_MLP_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
'Beta_MLP_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biasesBeta = {
'Beta_MLP_b1': tf.Variable(tf.random_normal([n_hidden_1])),
'Beta_MLP_b2': tf.Variable(tf.random_normal([n_hidden_2])),
'Beta_MLP_out': tf.Variable(tf.random_normal([n_classes]))
}  


def build_decoder(U, guesses):
    
    ##### select index #####
    def select(u, pos, idx): # u shape (1, 601, 400)
        u_idx = tf.gather(u, idx) #shape (601, 400)
        pos_idx = tf.gather(pos, idx)  #shape (1,)
        return tf.reshape(tf.gather(u_idx, pos_idx), [-1])   

    #######################
    
    n_input = 1600
    n_hidden_1 = hidden_size
    n_hidden_2 = hidden_size
    n_classes = 1
    
    
    ### two layer MLP ###
    def twoLayerMLPalpha(u_t, h, u_s, u_e):
        
        state_s = tf.concat([u_t, h, u_s, u_e],1)
        
        #n_input = state_s.get_shape()[1]
        n_input = 1600
        n_hidden_1 = hidden_size
        n_hidden_2 = hidden_size
        n_classes = 1    
        
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(state_s, weightsAlpha['Alpha_MLP_h1']), biasesAlpha['Alpha_MLP_b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weightsAlpha['Alpha_MLP_h2']), biasesAlpha['Alpha_MLP_b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_1, weightsAlpha['Alpha_MLP_out']) + biasesAlpha['Alpha_MLP_out']
        
        return out_layer
    #######################
    
    def twoLayerMLPbeta(u_t, h, u_s, u_e):
        
        state_s = tf.concat([u_t, h, u_s, u_e],1)
        
        #n_input = state_s.get_shape()[1]
        n_input = 1600
        n_hidden_1 = hidden_size
        n_hidden_2 = hidden_size
        n_classes = 1    
        
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(state_s, weightsBeta['Beta_MLP_h1']), biasesBeta['Beta_MLP_b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weightsBeta['Beta_MLP_h2']), biasesBeta['Beta_MLP_b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_1, weightsBeta['Beta_MLP_out']) + biasesBeta['Beta_MLP_out']
        
        return out_layer
    #######################
    
    
    with tf.variable_scope('selector'):
        #LSTM for decoding
        lstm_dec = tf.contrib.rnn.LSTMCell(hidden_size)
        
         # reshape self._u, (context, batch_size, 2*hidden_size)
        UU = tf.transpose(U[:,:max_timesteps,:], perm=[1, 0, 2]) # shape (600,1,400)
         # sample indices in one batch
        loop_until = tf.to_int32(np.array(range(batch_size)))
        
         # initial estimated positions
        s, e = tf.split(guesses,[1,1],1)
        
        fn = lambda idx: select(U, s, idx)
        u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32) # obtain u_s shape (1, 400)
        
        fn = lambda idx: select(U, e, idx)
        u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32) # obtain u_e shape (1, 400)
        
    s_seq, e_seq = [], []
    alpha_seq, beta_seq = [], []
    internal_h = (tf.zeros([batch_size, hidden_size]),tf.zeros([batch_size, hidden_size]))
    count = 0
    with tf.variable_scope('decoder') as vs:
        
        for step in range(max_decode_steps): # figure 3, the time steps, maximally 4 timesteps for iterative prediciton
            count+=1
            print count
            if step > 0:vs.reuse_variables() # once begin the second step, reuse all the previous parameters in this scope
            # single step lstm
            _input = tf.concat([u_s, u_e], 1) # shape (1, 800)
            _, h = tf.contrib.rnn.static_rnn(lstm_dec, [_input], internal_h, dtype=tf.float32) # h shape (1,200)
            internal_h = h
            h_state = tf.concat(h,1)  # shape (1, 400), combine c and h states together
            
            with tf.variable_scope('2layerMLP_alpha'): # replace the HMN 
                # compute start position next
                fn = lambda u_t: twoLayerMLPalpha(u_t, h_state, u_s, u_e) # u_t shape(1,400),one by one taken from U, see next line - kl
                alpha = tf.map_fn(lambda u_t: fn(u_t), UU, dtype=tf.float32)                                 
                s = tf.reshape(tf.argmax(alpha, 0), [batch_size]) # shape (1,) the guess of start index
                # update start guess
                fn = lambda idx: select(U, s, idx)
                u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32) # select out the guessed u_s shape (1,400)
                
            with tf.variable_scope('2layerMLP_beta'):
                # compute end position next
                fn = lambda u_t: twoLayerMLPbeta(u_t, h_state, u_s, u_e)
                beta = tf.map_fn(lambda u_t: fn(u_t), UU, dtype=tf.float32) # shape (600,1,1,1)
                e = tf.reshape(tf.argmax(beta, 0), [batch_size]) # shape (1,) the guess of end index
                # update end guess
                fn = lambda idx: select(U, e, idx)
                u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32) # select out the guessed u_e shape (1,400)
           
            s_seq.append(s)
            e_seq.append(e)
            alpha = tf.transpose(alpha,perm=[1,0,2])
            beta = tf.transpose(beta,perm=[1,0,2])
            alpha_seq.append(tf.reshape(alpha, [batch_size, -1])) # the final alpha scores for word position of each iteration, shape [[1,600],[1,600],[1,600],[1,600],[1,600]]
            beta_seq.append(tf.reshape(beta, [batch_size, -1])) # defined at line 208, shape [[1,600],[1,600],[1,600],[1,600],[1,600]]

            
    return alpha_seq, beta_seq, s_seq, e_seq



U = build_encoder(batch_context_ids, batch_question_ids)
alpha_seq,beta_seq,s_seq, e_seq = build_decoder(U, batch_guesses)

########################
## loss and optimizer ##
########################

def loss_shared(labels, logits):
    # might use gpu here
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,logits=logits,name='per_step_cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # tf.add_to_collection('per_step_losses', cross_entropy_mean)
    # return tf.add_n(tf.get_collection('per_step_losses'), name='per_step_loss')
    return cross_entropy_mean

print tf.shape(alpha_seq)
print tf.shape(batch_answer[:][0])

# definition of loss 
fn = lambda label, logit: loss_shared(label, logit)
loss_alpha = [fn(batch_answer[:,0],alpha) for alpha in alpha_seq]
loss_beta = [fn(batch_answer[:,1],beta) for beta in beta_seq]
loss = tf.reduce_sum([loss_alpha, loss_beta])/max_decode_steps


# optimizer
# use reserved gpu for gradient computation
#min_lr = 0.01
#lr = 0.1
# lr_rate = tf.maximum(
#         min_lr,
#         tf.train.exponential_decay(lr, self._global_step, 30000, 0.98))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# train accuracy
s_correct = tf.abs(s_seq[-1]-batch_answer[:,0])<=1
e_correct = tf.equal(tf.abs(e_seq[-1]-batch_answer[:,1]),0)
double_correct = s_correct & e_correct
sample_correct = tf.cast(double_correct,'float')
correct_pred = tf.reduce_mean(sample_correct)

# test accuracy is not calculated here, but use the prediction results to call
#  squad scripts to evaluate the F1 and EM scores
    

## for loop of batches learning ##
##################################

import time

epochs_num = 12
# max_training_steps = 3000
max_batch_number = int(all_question_ids.shape[0]/batch_size)
batch_number = 0
sample_count = 0 # global pointer to indicate which parts of the data should be in the new batch

# create next batch, recover the context_ids from the last column of question_ids 
def next_batch():
    
    global sample_count
    global batch_context_ids_data
    global batch_question_ids_data
    global batch_answer_data   
    
    # remove the samples with start or end index out of the maximum length of context
    for i in range(batch_size):
        # for epoches training, if the sample_count reach the end, then back to the start
        if sample_count>=all_question_ids.shape[0]:
            sample_count=0
        if all_answers[sample_count][1] < 600: 
            sample_index = all_question_ids[sample_count][-1] 
            batch_context_ids_data[i]=all_context_ids[sample_index]
            batch_question_ids_data[i]=all_question_ids[sample_count][:-1]# without the context id in the last column
            batch_answer_data[i] = all_answers[sample_count][:-1]# without the context id in the last column
            sample_count+=1
        else:
            sample_count+=1


init = tf.global_variables_initializer()
lossCollect = []

######################################
## for dev or test data loading#######
######################################

# create the dev data for evaluation
dev_batch_size = batch_size

dev_context_ids_data = np.zeros((dev_batch_size,max_timesteps),dtype=np.int)
dev_question_ids_data = np.zeros((dev_batch_size,min_timesteps),dtype = np.int)
dev_guesses_initial = np.zeros((dev_batch_size,2),dtype=np.int)

dev_sample_count = 0

def devData_next_batch():
    
    global dev_sample_count
    global dev_context_ids_data
    global dev_question_ids_data
       
    for i in range(dev_batch_size):
        
        if dev_sample_count>=dev_question_ids.shape[0]:
            dev_sample_count=0
        
        sample_index = dev_question_ids[dev_sample_count][-1] 
        dev_context_ids_data[i]=dev_context_ids[sample_index]
        dev_question_ids_data[i]=dev_question_ids[dev_sample_count][:-1]# without the context id in the last column
        dev_sample_count+=1
            
            
# Create a saver for writing training checkpoints
saver = tf.train.Saver(max_to_keep = epochs_num)

import sys

orig_stdout = sys.stdout
f = open('log721_1.txt', 'w')
sys.stdout = f

with tf.Session() as sess:
    # writer = tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init)
    start = time.time()
    for i in range(epochs_num):
#         lossEpoch = []
        train_accu = 0
        train_loss = 0
        while batch_number <= max_batch_number:
            next_batch()

            _,stepLoss, sLast, eLast,train_accuracy = sess.run([train_op,loss,s_seq[-1],e_seq[-1],correct_pred], 
                feed_dict={
                embeddingMatrix:embeddingCutLoad,
                batch_context_ids:batch_context_ids_data,
                batch_question_ids:batch_question_ids_data,
                batch_guesses: batch_guesses_initial,
                batch_answer: batch_answer_data
            })
            
#             if (batch_number%50 == 0):
#                 lossEpoch.append(stepLoss)
            
#             if(i>=0):
#                 if(batch_number>max_batch_number-40000):
#                     print('epoch #: ',i)
#                     print('batch_number: ', batch_number)             
#                     print('loss:',stepLoss)
#                     print('predictions: ', np.column_stack((sLast,eLast)))
#                     print('True answers: ', batch_answer_data)

            train_loss += stepLoss
    
            if(batch_number%800==0):
                print('train loss: ', stepLoss)
                print('train_accuracy: ', train_accuracy)
            
            train_accu += train_accuracy
            batch_number += 1    
            
        print('train epoch num: ', i)    
        print('train epoch loss: ', train_loss/batch_number)
        print('train epoch averaged accuracy: ', train_accu/batch_number)
        
        #print 'Saving'
        # saver.save(sess, '/home/liuk/coattention/model071/coattention_model_0715',global_step = i)

        batch_number=0
        
        #   saver.restore(sess, tf.train.latest_checkpoint('./'))
        dev_batch_count = 0
#         devLoss = 0
#         accuracy = 0
        #dev_pred = []
        while dev_sample_count < dev_size-dev_batch_size+1:
            devData_next_batch() 
            sLast_dev, eLast_dev =                    sess.run([s_seq[-1],e_seq[-1]],                                                      
                        feed_dict={
                            embeddingMatrix:embeddingCutLoad,
                            batch_context_ids:dev_context_ids_data,
                            batch_question_ids:dev_question_ids_data,
                            batch_guesses: dev_guesses_initial,                           
                        })
            
            #devLoss += batchDevLoss
            batch_pred = np.column_stack((sLast_dev, eLast_dev))
            if (dev_batch_count==0):
                epoch_pred = batch_pred
            else:
                epoch_pred = np.row_stack((epoch_pred,batch_pred))
            
#            print('dev_sample_count: ', dev_sample_count)
#             print('sLast_dev: ',sLast_dev)
#             print('true_s: ', dev_answer_data[:,0])
#             print('eLast_dev: ',eLast_dev)
#             print('true_e: ', dev_answer_data[:,1])
           
            #accuracy += correct_prediction
            dev_batch_count += 1
            
#             if (dev_batch_count%20==0):
#                 print('dev_batch_count: ', dev_batch_count)
#                 print('correct_predict: ', correct_prediction)
# #                 print('s_error: ', s_error)
# #                 print('e_error: ', e_error)
# #                 print('sample predict error:', sample_error)
#                 print('after every 20 batches pred, ground truth: ')
#                 print(np.column_stack((batch_pred, dev_answer_data)))  
#        #     print('acculumated accuracy: ', accuracy)
        epoch_pred = np.asarray(epoch_pred)
        np.save('/home/liuk/coattention/prediction/'+str(i)+'_0721', epoch_pred)
    #  dev_sample_count = 0
#         devLoss /= dev_batch_count
#         accuracy /= dev_batch_count
#         print('epoch #: ',i)
#         print('epoch Dev loss: ', devLoss)
#         print('epoch Dev accuracy: ', accuracy)        
#         print('epoch last batch pred, ground truth: ')
#         print(np.column_stack((batch_pred, dev_answer_data)))                         
        
        
    print('took sec time: ', time.time()-start)
    sys.stdout = orig_stdout
    f.close()  
  





