# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:28:27 2018

@author: ushasi2
link: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
"""

import UxUyLoader as uxuy
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
training_iters = 60 #200 
learning_rate = 0.001 
batch_size = 64
alpha = 0.01
l1 = 0.00001
l2 = 1
l3 = 1
t = 64

tf.reset_default_graph() 

dataset = uxuy.load_uxuy_dataset()
X = dataset[0]
Y = dataset[1]
L = dataset[2]

print("Training set (images X) shape: {shape}", X.shape)
print("Training set (images Y) shape: {shape}", Y.shape) 
print("Training set (labels) shape: {shape}", L.shape)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(L)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
L = onehot_encoder.fit_transform(integer_encoded)

p=int(X.shape[0]*0.96)
train_indices = np.random.choice(X.shape[0], p, replace=False)
train_X = X[train_indices,:]
train_Y = Y[train_indices,:]
train_L = L[train_indices,:]
test_indices = np.array(list(set(range(X.shape[0])) - set(train_indices)))
test_X = X[test_indices,:]
test_Y = Y[test_indices,:]
test_L = L[test_indices,:]


x = tf.placeholder("float", [None, 128])
y = tf.placeholder("float", [None, 128])
l = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


def load_model( sess, saver):
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

regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
weightsx = {
    #'wc1': tf.get_variable('W0', shape=(128,128), initializer=tf.contrib.layers.xavier_initializer()),
    #'wc2': tf.get_variable('W1', shape=(128,256), initializer=tf.contrib.layers.xavier_initializer()), 
    #'wc3': tf.get_variable('W2', shape=(256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1x': tf.get_variable('Wx3', shape=(128,t), regularizer = regularizer,initializer=tf.contrib.layers.xavier_initializer()),
    'wd2x': tf.get_variable('Wx4', shape=(t,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd3x': tf.get_variable('Wx5', shape=(256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'outx': tf.get_variable('Wx6', shape=(t,n_classes),regularizer = regularizer, initializer=tf.contrib.layers.xavier_initializer()), 
}
biasesx = {
    #'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    #'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    #'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1x': tf.get_variable('Bx3', shape=(t), regularizer = regularizer,initializer=tf.contrib.layers.xavier_initializer()),
    'bd2x': tf.get_variable('Bx4', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd3x': tf.get_variable('Bx5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'outx': tf.get_variable('Bx6', shape=(8), regularizer = regularizer,initializer=tf.contrib.layers.xavier_initializer()), #8
}

weightsy = {
    #'wc1': tf.get_variable('W0', shape=(128,128), initializer=tf.contrib.layers.xavier_initializer()),
    #'wc2': tf.get_variable('W1', shape=(128,256), initializer=tf.contrib.layers.xavier_initializer()), 
    #'wc3': tf.get_variable('W2', shape=(256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1y': tf.get_variable('Wy3', shape=(128,t), regularizer = regularizer, initializer=tf.contrib.layers.xavier_initializer()),
    'wd2y': tf.get_variable('Wy4', shape=(t,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd3y': tf.get_variable('Wy5', shape=(256,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'outy': tf.get_variable('Wy6', shape=(t,n_classes), regularizer = regularizer, initializer=tf.contrib.layers.xavier_initializer()), 
}
biasesy = {
    #'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    #'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    #'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1y': tf.get_variable('By3', shape=(t), regularizer = regularizer,initializer=tf.contrib.layers.xavier_initializer()),
    'bd2y': tf.get_variable('By4', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd3y': tf.get_variable('By5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'outy': tf.get_variable('By6', shape=(8), regularizer = regularizer,initializer=tf.contrib.layers.xavier_initializer()), #8
}

def fc_netx(x, weightsx, biasesx):  


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(x, [-1, weightsx['wd1x'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(x, weightsx['wd1x']), biasesx['bd1x'])
    fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weightsx['outx']), biasesx['outx'])

    fc2 = tf.reshape(fc1, [-1, weightsx['wd2x'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weightsx['wd2x']), biasesx['bd2x'])
    fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, keep_prob)
    return out, fc1, fc2

def fc_nety(y, weightsy, biasesy):  


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(y, [-1, weightsy['wd1y'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(y, weightsy['wd1y']), biasesy['bd1y'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weightsy['outy']), biasesy['outy'])

    fc2 = tf.reshape(fc1, [-1, weightsy['wd2y'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weightsy['wd2y']), biasesy['bd2y'])
    fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, keep_prob)
    return out,fc1, fc2



 
predx,fc_featsx,dux = fc_netx(x, weightsx, biasesx)
predy,fc_featsy,duy = fc_nety(y, weightsy, biasesy)



loss_reg= tf.losses.get_regularization_loss() 
''' 
unified = tf.matmul(x,fc_featsx)
cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(x,predx),labels=l)) #tf.nn.l2_loss(fc_featsx)#
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(y,predy), labels=l))
#tf.nn.l2_loss(fc_featsy)#
V = tf.reduce_mean(tf.norm(tf.subtract(tf.matmul(x,fc_featsx),tf.matmul(y,fc_featsy))))#X*w1 - Y*w1
'''

loss1 = tf.reduce_mean(tf.norm(tf.subtract(x,duy)))#tf.reduce_mean(tf.norm(fc_featsx))
loss2 = tf.reduce_mean(tf.norm(tf.subtract(y,dux)))#tf.reduce_mean(tf.norm(fc_featsy))#
loss3 = tf.reduce_mean(tf.norm(tf.subtract(fc_featsx,fc_featsy)))
loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predx,labels=l))
loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy, labels=l))
unified = fc_featsx


cost =  loss1 + loss2 + loss3 + 0*loss4 + 0*loss5 #l1*V + l2*cost1 + l3*cost2#loss_reg

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction1 = tf.equal(tf.argmax(predx, 1), tf.argmax(l, 1))
correct_prediction2 = tf.equal(tf.argmax(predy, 1), tf.argmax(l, 1))
#correct_prediction3 = tf.equal(tf.argmax(predx, 1), tf.argmax(l, 1))

#calculate accuracy across all the given images and average them out. 
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
accuracy = (accuracy1 + accuracy2)/2

init = tf.global_variables_initializer() 

#X = X - np.mean(X, axis=0)
#Y = Y - np.mean(Y, axis=0)

print('here 3')
with tf.Session() as sess:
    sess.run(init) 
    #print(pred.get_shape())
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    final_accuracy = 0
    path = 'srcV'
    saver = tf.train.Saver()
    model_name = 'unified'
    snapshot_path = path+'/snapshots/%s/' % (model_name)
    snapshot_path_latest2 = path+'/snapshots/%s/' % (model_name)
    latest2 = tf.train.latest_checkpoint(snapshot_path_latest2)
    
    #saver.restore(sess, latest2)
    #cur_i = int(latest2[len(snapshot_path_latest2 + 'model-'):])
    #print('Restoring last models default checkpoint at %d' % cur_i)
    summary_writer = tf.summary.FileWriter('./srcV/Output', sess.graph)
    print('here 4')
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))] 
            batch_l = train_L[batch*batch_size:min((batch+1)*batch_size,len(train_L))] 
            #batch_l = batch_l.reshape(len(batch_l),n_classes)
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, l: batch_l, keep_prob : 0.5})
            loss11,loss22, loss33, loss44, loss55, loss, acc = sess.run([loss1, loss2, loss3, loss4, loss5 ,cost, accuracy], feed_dict={x:batch_x,y: batch_y, l: batch_l, keep_prob : 1})

            print("Iter " + str(batch) + ", Loss1= " + "{:.2f}".format(loss11)+ ", Loss2= " + "{:.2f}".format(loss22)+ ", Loss3= " + "{:.2f}".format(loss33)+ ", Loss4= " + "{:.2f}".format(loss44)+ ", Loss5= " + "{:.2f}".format(loss55)+ ", Loss= " + "{:.2f}".format(loss) + ", Training Accuracy= " + "{:.2f}".format(acc))
        print("Iter " + str(i) + ", Loss= " + \

                      "{:.5f}".format(acc))

        print("Optimization Finished!")

        save_model(sess, saver, i)

        # Calculate accuracy for all 10000 mnist test images
	for batch in range(len(test_X)//batch_size):
	    batch_x = test_X[batch*batch_size:min((batch+1)*batch_size,len(test_X))]
            batch_y = test_Y[batch*batch_size:min((batch+1)*batch_size,len(test_Y))] 
            batch_l = test_L[batch*batch_size:min((batch+1)*batch_size,len(test_L))]

            test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: batch_x,y: batch_y,l : batch_l, keep_prob : 1})
            final_accuracy = final_accuracy + test_acc
            #print test_acc, batch

        #print (len(test_X)//batch_size)
	final_accuracy = final_accuracy/(len(test_X)//batch_size)

        train_loss.append(loss)

        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(final_accuracy))
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
                fc_featuresx = np.zeros((num,t)) # to store fully-connected layer's features
                fc_featuresy = np.zeros((num,t))
                #train_idx = train_indices # to store train index
                for ind in range(0,X.shape[0]/batch_size): #processing the datapoints batchwise
                    #reports = sess.run(self.reports, feed_dict={self.net.is_training:1})
                    ind_s = ind*batch_size
                    ind_e = (ind+1)*batch_size
		    batch_x = X[ind_s:ind_e,:]
                    batch_y = Y[ind_s:ind_e,:]
            	    batch_l = L[ind_s:ind_e,:] 
                    cfeatsx = sess.run(fc_featsx, feed_dict={x: batch_x, y:batch_y, l: batch_l, keep_prob : 1})
                    cfeatsy = sess.run(fc_featsy, feed_dict={x: batch_x, y:batch_y, l: batch_l, keep_prob : 1})
                    #print size(cpred)
                    #print cpred.shape()
                    fc_featuresx[ind_s:ind_e,:] = cfeatsx
                    fc_featuresy[ind_s:ind_e,:] = cfeatsy
                    #print('Processing %d batch' % ind)
                # storing the extracted features in mat file
                sio.savemat('unified_features.mat', {'featuresx':fc_featuresx, 'featuresy':fc_featuresy}) #saving


