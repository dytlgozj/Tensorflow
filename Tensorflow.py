# tensorflow 를 사용하려고 먼저 import한다.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
# x를 [None, 784]형태의 부정소숫점으로 이루어진 2차원 텐서로 표현한다.
x = tf.placeholder(tf.float32, [None, 784])
# w와 b는 학습할 변수들이다.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# y는 matmul의 표현식으로 x와 W를 곱한다. 그 다음 b를 더하고 마지막으로 tf.nn.softmax를 적용한다.
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 교차 엔트로피를 구하기 위해 새로운 placeholder를 정의한다.
y_ = tf.placeholder(tf.float32, [None, 10])
# 그 다음 교차 엔트로피를 구한다.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 경사 하강법을 이용하여 교차엔트로피를 최소화하는 코드이다.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
# 마지막으로 만든 변수들을 모두 초기화하는 과정
init = tf.initialize_all_variables()
# 세션에서 모델을 시작하고 변수들을 초기화하는 작업
sess = tf.Session()
sess.run(init)

# Learning
# 학습을 1000번 시킨다.
for i in range(1000):
  # 임의로 학습세트로부터 100개의 무작위 데이터의 일괄 처리들을 가져온다.
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # 그 다음 placeholder를 대체하기 위한 일괄 처리 데이터에 train_step 피딩을 실행한다.
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation
# tf.equal을 이용해 예측이 실제와 맞는지 확인한다.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 부울 리스트인데, 부동 소숫점으로 캐스팅한 후 평균값을 구한다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
# 테스트 데이터를 대상으로 정확도를 확인하는 과정이다. 그것을 print함.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
