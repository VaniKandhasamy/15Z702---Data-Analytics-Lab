
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sbs


# In[17]:


trainDf = pd.read_csv('train.csv').dropna()
testDf = pd.read_csv('test.csv')


# In[18]:


trainDf.plot.scatter('x', 'y', s=1)


# In[19]:


sess = tf.Session()


# In[20]:


x = tf.placeholder( tf.float64, shape=(1, None), name='x')
y = tf.placeholder( tf.float64, shape=(1, None), name='y')
w = tf.Variable(tf.random_normal((1,1), dtype=tf.float64))
b = tf.Variable(tf.random_normal((1,1), dtype=tf.float64))
y_hat = tf.matmul(w, x) + b
loss = tf.reduce_sum(tf.pow(y_hat - y, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000002)
optimizer_node = optimizer.minimize(loss)
initializer = tf.global_variables_initializer()


#  # RUNNING THE SESSION 

# In[21]:


sess.run(initializer)
w_val, b_val, loss_val = sess.run([w, b, loss], feed_dict = {x: trainDf['x'].values.reshape(1, -1), y: trainDf['y'].values.reshape(1, -1)})
print(w_val, b_val, loss_val)
for _ in range(100):
    sess.run([optimizer_node], feed_dict = {x: trainDf['x'].values.reshape(1, -1), y: trainDf['y'].values.reshape(1, -1)})
    w_val, b_val, loss_val = sess.run([w, b, loss], feed_dict = {x: trainDf['x'].values.reshape(1, -1), y: trainDf['y'].values.reshape(1, -1)})
    print(w_val, b_val, loss_val)


# # Finding the linear line

# In[24]:


w_val, b_val = sess.run([w, b])
x_sweep = np.linspace(0, 100, 20)
y_sweep = w_val[0] * x_sweep + b_val[0]
plt.scatter(trainDf['x'], trainDf['y'], s=1)
plt.plot(x_sweep, y_sweep, 'r')


# In[25]:


w_val, b_val, loss_val = sess.run([w, b, loss], feed_dict = {x: testDf['x'].values.reshape(1, -1), y: testDf['y'].values.reshape(1, -1)})
print(w_val, b_val, loss_val)

