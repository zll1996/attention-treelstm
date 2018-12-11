import numpy as np
import tensorflow as tf
import os
import math
import matplotlib as plt
from os import listdir
from os.path import isfile, join
import re
from random import randint
import datetime

wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

print(len(wordsList))
print(wordVectors.shape)

reviewFiles = ['pos_2.txt', 'neg_2.txt']
numWords = []
for f in reviewFiles:
    with open(f, "r", encoding='utf-8') as f:
        for line in f:
            counter = len(line.split())
            numWords.append(counter)       
print('files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

maxSeqLength = 40

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
lineCounter = 0

for file in reviewFiles:  
    with open(file, "r") as f:
        for line in f:
            indexCounter = 0
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[lineCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[lineCounter][indexCounter] = 399999 #Vector for unkown words
                indexCounter = indexCounter + 1
                
                if indexCounter >= maxSeqLength:
                    break
            lineCounter = lineCounter + 1 

#Pass into embedding function and see if it evaluates. 

np.save('idsMatrix', ids)

ids = np.load('idsMatrix.npy')


def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,4462)
            labels.append([1,0])
        else:
            num = randint(5462,9610)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(4462,5462)
        if (num <= 4962):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

batchSize = 32
lstmUnits = 64
numClasses = 2
iterations = 50001
numDimensions = 300

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

   #Save the network every 10,000 training iterations
   if (i % 1000 == 0 and i != 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

iterations = 10
acc = []
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    tmp = (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100
    print("Accuracy for this batch:", tmp)
    acc.append(tmp)

print("Overall accuracy:", sum(acc)/len(acc))

