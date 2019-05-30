import tensorflow as tf

'''
변수 4개를 통해 나오는 1개 결과는 (0, 1, 2)세가지 중 하나. 
y_data 는 one-hot encoding 방식으로 표현된 결과 

'''
#data
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0] ]
#y_data = [[2], [2], [2], [1],
nb_classes = 3
#node
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([1, nb_classes]), name='bias')

#행렬곱으로 나온 실수로 이루어진 3행1열 배열에 소프트맥스 값을 사용하여 확률로이루어진 3행1열 배열.

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

proc1 = -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)
proc2 = tf.reduce_mean(proc1)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
#cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, lables=y_data)
#cost = tf.reduce_mean(cost_i)
"""
과정
hypothesis를 실행했을 때 Nx3 행렬. y_data의 형태. axis=1 : 2차원 제거하면서 1차원에 합한다. 
Y * -log(hypothesis)는 가설이 올바를 때 0에 수렴. 
y_data가 one-hot형태이기 때문에 binary classification의 0과 1을 모두 확인할 수 있음. 
multi classification은 binary classification의 조합. 
"""
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, h_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
        proc1_val, proc2_val, W_val= sess.run([proc1, proc2, W], feed_dict={X:x_data, Y:y_data})
        if step%200==0:
            #print('step: {0}, hypothesis: {1}, cost: {2}'.format(step, h_val, cost_val))
            print("proc1 = {}\n, proc2 = {}\n, hypothesis = {}\n, W = {}\n".format(proc1_val, proc2_val, h_val, W_val))