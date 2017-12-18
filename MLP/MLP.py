import tensorflow as tf
import numpy as np
import matplotlib.pylot as plt
#import cPickle as pickle

#import seaborn


#hyperparameters
learning_rate = 0.001
training_epochs = 500
batch_size = 500
display_step = 10
n_hidden_1 = 256
n_hidden_2 = 128
n_hidden_3 = 64

# sample statistics
n_sample = XData.shape[0]    
n_input = XData.shape[1]
n_class = YData.shape[1]
  
weight = {
    'h1': tf.Variable( tf.random_normal([n_input, n_hidden_1]) ),
    'h2': tf.Variable( tf.random_normal([n_hidden_1, n_hidden_2]) ),
    'h3': tf.Variable( tf.random_normal([n_hidden_2, n_hidden_3]) ),
    'h4': tf.Variable( tf.random_normal([n_hidden_3, n_class]) )
}

bias = {
    'h1': tf.Variable( tf.random_normal([n_hidden_1]) ),
    'h2': tf.Variable( tf.random_normal([n_hidden_2]) ),
    'h3': tf.Variable( tf.random_normal([n_hidden_3]) ),
    'h4': tf.Variable( tf.random_normal([n_class]) )
}


def mlp_forward(x, weight, bias):
    
    
    Z1 = tf.add( tf.matmul(x, weight['h1']), bias['h1'] )
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.add( tf.matmul(A1, weight['h2']), bias['h2'])
    A2 = tf.nn.relu(Z2)
    
    Z3 = tf.add( tf.matmul(A2, weight['h3']), bias['h3'])
    A3 = tf.nn.relu(Z3)
    
    Z4 = tf.add( tf.matmul(A3, weight['h4']), bias['h4'])
    A4 = tf.nn.softmax(Z4)
    
    return A4
    

def mlp_train(XData, YData, weight, bias, 
              n_sample, n_input, n_class,
              learning_rate, training_epochs, 
              batch_size, 
              ):
    
    
    #varibles
     = get_param(XData, XData)
    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_class])    
    
    #forward propagation
    pred = mlp_forward(x, weight, bias)
    
    #cost function
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(pred, y) )
    
    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    #initilize
    init = tf.initialize_all_variables()
    
    correct_prediction = tf.equal( tf.argmax(pred, 1), tf.argmax(y, 1) )
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, 'float') )
    
    #train
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_sample/batch_size)
            
            for i in range(total_batch):
                _, c = sess.run([optimizer, cost], feed_dict={x: XData[i*batch_size:(i+1)*batch_size, :],
                                                              y: YData[i*batch_size:(i+1)*batch_size, :]})
                avg_cost += c / total_batch
            
            plt.plot(epoch+1, avg_cost)
            
            if epoch % display_step == 0:
                print('Epoch:', '%04d'% (epoch + 1), 'cost=', '{:,9f}'.format(avg_cost))
    
    print 'Model Training Finished!'
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    