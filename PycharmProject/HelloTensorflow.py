import tensorflow as tf

#콘스탄트라는 노드 생성. 노드 이름은 hi, 노드 데이터는 헬로 텐서플로!
hi = tf.constant("hello! tensorflow!")

#세션 생성
sess = tf.Session()

#세션을 통해 hi라는 노드 실행
print(sess.run(hi))
print(str(sess.run(hi), encoding='utf-8'))

#두 노드를 만들어 하나로 합하는 그래프 생성
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) #also tf.float32 implicitly
node3 = tf.add(node1, node2)

#출력
print("node1 : ", node1, "node2 : ", node2)
print("node3 : ", node3)
#결과값이 나오지 않음. 단지 세 개의 텐서가 존재한다고 알려줌

#결과값을 확인하기 위해서는 세션을 통해 실행시켜야 한다.
sess = tf.Session()
print("sess.run([node1, node2]) : {0}".format(sess.run([node1, node2])))
print("sess.run(node3) : ", sess.run(node3))

#실행시키는 과정에서 값을 넣을 때
node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
adder_node = node1 + node2

print(sess.run(adder_node, feed_dict={node1:3, node2:4.5}))
print(sess.run(adder_node, feed_dict={node1:[1,3], node2:[2,4]}))
#실행 도중에 값을 수정한다는게 중요한 것 같다.