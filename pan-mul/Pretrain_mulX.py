# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:28:27 2018

@author: ushasi2
link: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
"""

import mspan as sc
import LabelCMFHashing as lf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.io as sio
import numpy as np
import tensorflow as tf
import os

k = 128 # k-bits of hash code
lamda = 0.01
mu = 0.01
chi = 0.01
n_classes = 8
training_iters = 1 #200 
learning_rate = 0.001 
batch_size = 128
alpha = 0.01

tf.reset_default_graph() 

dataset = sc.load_mspan_dataset()
X = dataset[0]
Y = dataset[1]
L = dataset[2]

p=int(X.shape[0]*0.98)
train_indices = np.random.choice(X.shape[0], p, replace=False)
#train_indices = train_indices.astype(int)
train_X = X[train_indices,:,:,:]
#train_Y = Y[train_indices,:,:]
train_L = L[train_indices]
test_indices = np.array(list(set(range(X.shape[0])) - set(train_indices)))
test_X = X[test_indices,:,:,:]
#test_Y = Y[test_indices,:,:]
test_L = L[test_indices]
#both placeholders are of type float
X = X.reshape(-1, 64, 64, 4)
train_X = train_X.reshape(-1, 64, 64, 4)
test_X = test_X.reshape(-1,64,64,4)
#train_Y = train_Y.reshape(-1, 256,256, 1)
#test_Y = test_Y.reshape(-1, 256,256, 1)

print('here 1')
x = tf.placeholder("float", [None, 64,64,4])
#y = tf.placeholder("float", [None, 256,256,1])
l = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,4,32), initializer=tf.contrib.layers.xavier_initializer()), #(3,3,4,32)
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(8), initializer=tf.contrib.layers.xavier_initializer()), #8
}
def lrelu(x):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return lrelu(x) #leaky_relu 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

 

def conv_netx(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #conv1 = batch_norm(conv1, 32)
    conv1 = tf.layers.batch_normalization(conv1,momentum=0.99,epsilon=0.001, center=True,scale=True,
    training=False,trainable=True)

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #conv2 = batch_norm(conv2, 64)
    conv2 = tf.layers.batch_normalization(conv2,momentum=0.99,epsilon=0.001, center=True,scale=True,
    training=False,trainable=True)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    #conv3 = batch_norm(conv3, 128)
    conv3 = tf.layers.batch_normalization(conv3,momentum=0.99,epsilon=0.001, center=True,scale=True,
    training=False,trainable=True)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out,fc1

def load_model(sess, saver):
        latest = tf.train.latest_checkpoint(snapshot_path)
        print(latest)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(snapshot_path + 'model-'):])
        print("Model restored at %d." % i)
        return i
        
def save_model( sess, saver, i):
        if not os.path.exists(snapshot_path):
                os.makedirs(snapshot_path)
        latest = tf.train.latest_checkpoint(snapshot_path)
        #if latest == None or i != int(latest[len(self.snapshot_path + 'model-'):]):
        if 1:
            print('Saving model at %d' % i)
            #verify_dir_exists(self.snapshot_path)
            result = saver.save(sess, snapshot_path + 'model', global_step=i)
            print('Model saved to %s' % result)



pred,fc_feats = conv_netx(x, weights, biases)
#predy = conv_nety(x,y, weightsy, biasesy)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=l))
#cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy, labels=l))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(l, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()





print('here 2')
size1 = np.shape(X)
size2 = np.shape(Y)
#X = np.reshape(X,(len(X), size1[1]*size1[2]*size1[3]))
#Y = np.reshape(Y,(len(Y), size2[1]*size2[2]))

size1 = np.shape(X)
size2 = np.shape(Y)
#X = X - np.mean(X, axis=0)
#Y = Y - np.mean(Y, axis=0)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(L)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
L = onehot_encoder.fit_transform(integer_encoded)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train_L)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
train_L = onehot_encoder.fit_transform(integer_encoded)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(test_L)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
test_L = onehot_encoder.fit_transform(integer_encoded)
#train_L = train_L.transpose()
#X = X.transpose()
#Y = Y.transpose()
'''
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
}




#zero centering X, Y and L


#ux and uy define
H = np.zeros((len(X),k))
Ux = np.zeros((size1[1],k))
Uy = np.zeros((size2[1],k))


#px and py define
Px = np.zeros((k,size1[1]))
Py = np.zeros((k,size2[1]))

#V matrix
V = np.zeros((k,len(X)))

#output = lf.LCMFH(X,Y,L,Px,Py,Ux,Uy,V,H,k,lamda,mu,chi)
'''
fc_features = None
print('here 3')
with tf.Session() as sess:
    sess.run(init) 
    print(pred.get_shape())
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    path = 'src'
    saver = tf.train.Saver()
    model_name = 'multispectral'
    snapshot_path = path+'/snapshots/%s/' % (model_name)
    snapshot_path_latest2 = path+'/snapshots/%s/' % (model_name)
    latest2 = tf.train.latest_checkpoint(snapshot_path_latest2)
    
    saver.restore(sess, latest2)
    cur_i = int(latest2[len(snapshot_path_latest2 + 'model-'):])
    print('Restoring last models default checkpoint at %d' % cur_i)
    summary_writer = tf.summary.FileWriter('./src/Output', sess.graph)
    print('here 4')
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            #batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))] 
            batch_l = train_L[batch*batch_size:min((batch+1)*batch_size,len(train_L))] 
            #batch_l = batch_l.reshape(len(batch_l),n_classes)
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, l: batch_l, keep_prob : 0.5})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              l: batch_l, keep_prob : 1})
            #print("Iter " + str(batch) + ", Loss= " + \"{:.6f}".format(loss) + ", Training Accuracy= " + \"{:.5f}".format(acc))
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")
        save_model(sess, saver, i)
        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,l : test_L, keep_prob : 1})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()

fc_features = None
print('Model "%s" already trained!'% model_name)
with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                #sess.run(tf.local_variables_initializer(), variable_initialization)
                print('Starting threads')
                saver = tf.train.Saver()  # Gets all variables in `graph`.
                i = load_model(sess, saver)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                num = 80000
                fc_features = np.zeros((num,128)) # to store fully-connected layer's features
                #train_idx = train_indices # to store train index
                for ind in range(0,X.shape[0]/batch_size): #processing the datapoints batchwise
                    #reports = sess.run(self.reports, feed_dict={self.net.is_training:1})
                    ind_s = ind*batch_size
                    ind_e = (ind+1)*batch_size
		    batch_x = X[ind_s:ind_e,:,:,:]
                    #batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))] 
            	    batch_l = L[ind_s:ind_e,:] 
                    cfeats = sess.run(fc_feats, feed_dict={x: batch_x, l: batch_l, keep_prob : 1})
                    
                    #print size(cpred)
                    #print cpred.shape()
                    fc_features[ind_s:ind_e,:] = cfeats
                    #print('Processing %d batch' % ind)
                # storing the extracted features in mat file
                sio.savemat('mul_features.mat', {'features':fc_features}) #saving






















