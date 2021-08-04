import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class SVR(object):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        
    def fit(self, X, y, length = 100, rate=0.1):
        self.sess = tf.compat.v1.Session()
        
        feature = X.shape[-1] if len(X.shape) > 1 else 1
        
        self.X = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)
        
        self.b = tf.Variable(tf.random_normal(shape=(1,)))
        self.W = tf.Variable(tf.random_normal(shape=(feature, 1)))
                
        self.y_pred = tf.matmul(self.X, self.W) + self.b
        
        self.loss = tf.reduce_mean(tf.maximum(0., tf.abs(self.y_pred - self.y) - self.epsilon)) + tf.norm(self.W)/2

        
        opt = tf.train.GradientDescentOptimizer(learning_rate=rate)
        opt_op = opt.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        
        for i in range(length):
            
            self.sess.run(
                opt_op, 
                {
                    self.X: X,
                    self.y: y
                }
            )
            
        return self
            
    def predict(self, X, y=None):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        y_pred = self.sess.run(
            self.y_pred, 
            {
                self.X: X 
            }
        )
        return y_pred