import numpy as np
import statistics
import pandas as pd
# import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
# from os import walk
import os, glob
# %matplotlib inline
# import numpy as np

def create_data(dir):
    pixels = []
    hemmor_labels = []
    # nonhemmor_labels = []

    height  = 64
    width = 64
    for infile in glob.glob("./"+dir+"/hemmorhage_data/*.png"):
        img =  Image.open(infile)
        img = img.resize((width, height), Image.ANTIALIAS).convert("RGB")
#         img = img.load
        img = np.asarray(img)/255.0
        img = img.tolist()
        pixels.append(img)
        hemmor_labels.append(1)
        # nonhemmor_labels.append(0)

    for infile in glob.glob("./"+dir+"/non_hemmorhage_data/*.png"):
        img =  Image.open(infile)
        img = img.resize((width, height), Image.ANTIALIAS).convert("RGB")
#         img = img.load()
        img = np.asarray(img)/255.0
        img = img.tolist()
        pixels.append(img)
        # nonhemmor_labels.append(1)
        hemmor_labels.append(0)

    data = {"pixels" : pixels, "hemmorhage" :hemmor_labels }
    # hemmors = {"pixels":data_hemmor_pixels, "label" :hemmor_labels}
    # nonhemmors = {"pixels":data_nonhemmor_pixels, "label" :nonhemmor_labels}
    #
    # data1 = pd.DataFrame(hemmors)
    # data2 = pd.DataFrame(nonhemmors)
    # data = pd.concat([data1,data2])
    data = pd.DataFrame(data)
    data = data.sample(frac=1).reset_index(drop=True)
    # data = np.array(data.to_numpy)
    return data

def placeXY():
    X = tf.placeholder(tf.float32 , shape=[None,64,64,3])
    y = tf.placeholder(tf.float32, shape=[None,1])
    return X,y

learning_rate = tf.placeholder(tf.float32 , shape=[])
epochs = 125

batch_size = 35
losses = []
testLosses = []
accuracies = []
_, ax = plt.subplots()

def fit(train,test,optimizer,cross_entropy,X,y,yclipped) :
    xtest =  test["pixels"]
    ytest =  test[["hemmorhage"]]
    with tf.compat.v1.Session() as sess :
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        # lr = 0.01
        # test_loss = 0

        for epoch in range(epochs):
            avg_cost = 0
            # if (epoch >= 40) :
                # lr = 0.001

            for chunk in np.array_split(train, len(train["hemmorhage"])/batch_size):
                xb = chunk["pixels"]
                xb = np.array(xb.to_numpy().tolist())
                # print(xb.shape)
                yb = chunk[["hemmorhage"]]

                # print (xb)
                (_, c) = sess.run([optimizer, cross_entropy],feed_dict={X : xb ,y: yb.to_numpy()})
                avg_cost += c / batch_size


            test_loss,yc,yt = sess.run( [cross_entropy,yclipped,y],feed_dict={X : np.array(xtest.to_numpy().tolist()) ,y :ytest.to_numpy()})
            losses.append(avg_cost)
            testLosses.append(test_loss)
            accur = acc(yc,yt)
            print(accur)
            accuracies.append(accur)
            # print(sess.run(, feed_dict={X:np.array(xtest.to_numpy().tolist()), y:ytest.to_numpy()}))
            yc, yt = sess.run([yclipped,y], feed_dict={X:np.array(train["pixels"].to_numpy().tolist()), y:train[["hemmorhage"]].to_numpy()})
            print(acc(yc,yt))
            # print)
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost),"testloss : ","{:.3f}".format(test_loss))


def plots():
    plt.plot(losses ,label="losses")
    plt.plot(testLosses ,label="testLosses")
    plt.plot(accuracies ,label="accuracy")
    legend = ax.legend(loc='upper center', shadow=True, fontsize='large')

    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')
    plt.show()
# def
def acc (yc,yt):
    arr = []
    for i in range(len(yc)) :
        x = 0
        if yc[i][0]  >= 0.5 :
            x = 1
        else :
            x = 0
        if yt[i][0] == x :
            arr.append(1)
        else :
            arr.append(0)
    return statistics.mean(arr)

def setFeedForward(X):
    tf.compat.v1.set_random_seed(42)
    weights = []
    W1 = tf.compat.v1.get_variable(name="W1",shape=[3,3,3,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.compat.v1.get_variable(name="W2",shape=[3,3,1,32],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.compat.v1.get_variable(name="W3",shape=[3,3,1,64],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # W4 = tf.compat.v1.get_variable(name="W4",shape=[3,3,1,512],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W9 = tf.Variable(tf.random.normal([4096, 2000], stddev=2/1000), name='W9')
    b9 = tf.Variable(tf.random.normal([2000]), name='b9')
    W10 = tf.Variable(tf.random.normal([2000, 400], stddev=2/1000), name='W10')
    b10 = tf.Variable(tf.random.normal([400]), name='b10')
    W11 = tf.Variable(tf.random.normal([400, 1], stddev=2/1000), name='W11')
    b11 = tf.Variable(tf.random.normal([1]), name='b11')
    weights.extend((W10,W11,W9))
#     xshapped = tf.reshape(X, [-1, 28, 28, 1])
    conv1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool2d(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    conv2 = tf.nn.conv2d(pool1,W2,strides=[1,1,1,1],padding="SAME" )
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool2d(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    conv3 = tf.nn.conv2d(pool2,W3,strides=[1,1,1,1],padding="SAME" )
    conv3 = tf.nn.relu(conv3)
    pool3 = tf.nn.max_pool2d(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    # conv4 = tf.nn.conv2d(pool3,W4,strides=[1,1,1,1],padding="SAME" )
    # conv4 = tf.nn.relu(conv4)
    # pool4 = tf.nn.max_pool2d(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    pool3flattened = tf.contrib.layers.flatten(pool3)

    dense3 = tf.add(tf.matmul(pool3flattened,W9),b9)
    dense3 = tf.nn.relu(dense3)

    dense4 = tf.add(tf.matmul(dense3,W10),b10)
    dense4 = tf.nn.relu(dense4)

    y_ = tf.nn.sigmoid(tf.add(tf.matmul(dense4,W11),b11))
    # yclipped =  tf.clip_by_value(y_, 1e-10, 0.9999999)
    return y_,weights



def cross_ent(yclipped,y,weights):
    # cross_entropy = -tf.reduce_mean(y * tf.log(yclipped)+ (1 - y) * tf.log(1 - yclipped))
    print(tf.shape(yclipped),">>>>>>>>>>>>>>>>>>>>>>>>>>")
    regularizers = 0
    for i in range(len(weights)) :
        regularizers += tf.nn.l2_loss(weights[i])

    # cross_entropy = -tf.reduce_mean(tf.reduce_sum((y * tf.log(yclipped)+ (1 - y) * tf.log(1 - yclipped)),axis = 1))
    loss = tf.reduce_mean(-(y * tf.log(yclipped)+ (1 - y) * tf.log(1 - yclipped))+ 0.01 *regularizers)
    return loss

# def
def cnntumors():
    X,y = placeXY()
    yclipped,weights = setFeedForward(X)
    cost = cross_ent(yclipped,y,weights)
    train = create_data("training_set")
    test = create_data("test_set")
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    # yclipped  = yc/
    # correct_prediction = tf.equal(y, yclipped)

     # = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    fit(train,test,optimizer , cost,X,y,yclipped)
    plots()
cnntumors()
# True False False False False True True False False False False False True True True True True True False True
# True False False False False True True False False False False False True True True True True True False True
