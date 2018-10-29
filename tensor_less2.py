import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

n_node_hl1 = 500
n_node_hl2 = 500
n_node_hl3 = 500
n_classes = 10
batch_size = 100
# let's define the placeholder in tensorflow
x = tf.placeholder('float',[None,784])
# the second parameter is the placeholder
y = tf.placeholder('float')

# define the neraul network model

def neural_network_model(data):
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784,n_node_hl1])), 'biases': tf.Variable(tf.random_normal([n_node_hl1]))}
	
	
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_node_hl1,n_node_hl2])), 'biases': tf.Variable(tf.random_normal([n_node_hl2]))}
	
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_node_hl2,n_node_hl3])), 'biases': tf.Variable(tf.random_normal([n_node_hl3]))}
	
	output_layer  =  {'weights':tf.Variable(tf.random_normal([n_node_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
	
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
	
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
	
	l3 = tf.nn.relu(l3)
	out = tf.matmul(l3,output_layer ['weights']) + output_layer ['biases']
	return out

# we started here to run our model
def run_train_the_neural_network(x):
	prediction = neural_network_model(x)
	
	# measure how much loss when we using the neural network
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
	
	with tf.Session() as sess:
		# init the variable into session of tensorflow
		sess.run(tf.global_variables_initializer())

		hm_epoch = 10
		for e in range(hm_epoch):
			# the training data sets devided by the batch size
			e_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x , epoch_y = mnist.train.next_batch(batch_size)
				_ , c = sess.run([optimizer, cost],feed_dict={x:epoch_x,y:epoch_y})
				e_loss += c
				print ('epoch:',e,'completed of ',hm_epoch,'loss:',e_loss)
		crr = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		arr = tf.reduce_mean(tf.cast(crr, 'float'))
		print('Accuracy:',arr.eval({x:mnist.test.images, y:mnist.test.labels}))
				
run_train_the_neural_network(x)
# end run 	