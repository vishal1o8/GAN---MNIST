import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
print("Import")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
#This function performns a leaky relu activation, which is needed for the discriminator network.
print("loaded MNIST datasets")
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
    
#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img
def generator(z):
    
    zP = slim.fully_connected(z,4*4*256,normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
    zCon = tf.reshape(zP,[-1,4,4,256])
    
    gen1 = slim.convolution2d_transpose(\
        zCon,num_outputs=64,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv1', weights_initializer=initializer)
    
    gen2 = slim.convolution2d_transpose(\
        gen1,num_outputs=32,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv2', weights_initializer=initializer)
    
    gen3 = slim.convolution2d_transpose(\
        gen2,num_outputs=16,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv3', weights_initializer=initializer)
    
    g_out = slim.convolution2d_transpose(\
        gen3,num_outputs=1,kernel_size=[32,32],padding="SAME",\
        biases_initializer=None,activation_fn=tf.nn.tanh,\
        scope='g_out', weights_initializer=initializer)
    
    return g_out
def discriminator(bottom, reuse=False):
    
    dis1 = slim.convolution2d(bottom,16,[4,4],stride=[2,2],padding="SAME",\
        biases_initializer=None,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv1',weights_initializer=initializer)
    
    dis2 = slim.convolution2d(dis1,32,[4,4],stride=[2,2],padding="SAME",\
        normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv2', weights_initializer=initializer)
    
    dis3 = slim.convolution2d(dis2,64,[4,4],stride=[2,2],padding="SAME",\
        normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv3',weights_initializer=initializer)
    
    d_out = slim.fully_connected(slim.flatten(dis3),1,activation_fn=tf.nn.sigmoid,\
        reuse=reuse,scope='d_out', weights_initializer=initializer)
    
    return d_out


tf.reset_default_graph()

z_size = 100 #Size of z vector used for generator.

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

#These two placeholders are used for input into the generator and discriminator, respectively.
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) #Random vector
real_in = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32) #Real images

Gz = generator(z_in) #Generates images from random z vectors
Dx = discriminator(real_in) #Produces probabilities for real images
Dg = discriminator(Gz,reuse=True) #Produces probabilities for generator images

#These functions together define the optimization objective of the GAN.
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.

tvars = tf.trainable_variables()
# print("trainable_variables()")
# print(len(tvars))
#The below code is responsible for applying gradient descent to update the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #Only update the weights for the discriminator network.
g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #Only update the weights for the generator network.

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)



batch_size = 128 #Size of image batch to apply at each iteration.
iterations = 500000 #Total number of iterations to use.
sample_directory = './figs' #Directory to save sample images from generator in.
model_directory = './models' #Directory to save trained model to.

init = tf.initialize_all_variables()
saver = tf.train.Saver()
import cv2
with tf.Session() as sess:  
    sess.run(init)
    for i in range(iterations):
        zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch
        xs,_ = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
        # x0 = xs[1]

        # print(np.reshape(np.array(x0),[28,28]))
        # cv2.imshow("lo",np.reshape(np.array(x0),[28,28]))
        # cv2.waitKey(500)
        
        xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
        xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
        _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs,real_in:xs}) #Update the discriminator
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs}) #Update the generator, twice for good measure.
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs})
        if i % 10 == 0:
            print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)

            z2_1 = np.random.uniform(-1.0,1.0,size=[batch_size/3,z_size]).astype(np.float32) #Generate another z batch
            z2_2 = np.random.uniform(-1.0,1.0,size=[batch_size/3,z_size]).astype(np.float32) #Generate another z batch
            z2_3 = np.random.uniform(-1.0,1.0,size=[batch_size/3,z_size]).astype(np.float32) #Generate another z batch
            # z2_4 = np.random.uniform(-1.0,1.0,size=[batch_size/5,z_size]).astype(np.float32) #Generate another z batch
            # z2_5 = np.random.uniform(-1.0,1.0,size=[batch_size/5,z_size]).astype(np.float32) #Generate another z batch
            
            newZ1 = sess.run(Gz,feed_dict={z_in:z2_1}) #Use new z to get sample images from generator.
            newZ2 = sess.run(Gz,feed_dict={z_in:z2_2}) #Use new z to get sample images from generator.
            newZ3 = sess.run(Gz,feed_dict={z_in:z2_3}) #Use new z to get sample images from generator.
            # newZ4 = sess.run(Gz,feed_dict={z_in:z2_4}) #Use new z to get sample images from generator.
            # newZ5 = sess.run(Gz,feed_dict={z_in:z2_5}) #Use new z to get sample images from generator.
            
            newZ = np.array(list(newZ1)+list(newZ2)+list(newZ3))

            newZ = newZ[np.random.permutation(len(newZ))]            
            # newX = sess.run(Dg, feed_dict={z_in:z2,real_in:newZ})
            # newX2 = sess.run(Dx, feed_dict={real_in:newZ})

            # print(len(newZ))

            # print(len(newZ),len(newZ[0]))
            # for i in range(len(newZ)):
	           #  cv2.imshow(str(newX2[i]),np.reshape(newZ[i],[32,32]))
	           #  print(str(newX2[i]))
	           #  cv2.waitKey(50)
            
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            #Save sample generator images for viewing training progress.
            for j in xrange(len(newZ)/64):
            	save_images(np.reshape(newZ[64*j:64*(j+1)],[64,32,32]),[8,8],sample_directory+'/fig'+str(i)+'-'+str(j)+'.png')
        if i % 1000 == 0 and i != 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
            print "Saved Model"

            

# sample_directory = './figs' #Directory to save sample images from generator in.
# model_directory = './models' #Directory to load trained model from.
# batch_size_sample = 36

# init = tf.initialize_all_variables()
# saver = tf.train.Saver()
# with tf.Session() as sess:  
#     sess.run(init)
#     #Reload the model.
#     print 'Loading Model...'
#     ckpt = tf.train.get_checkpoint_state(model_directory)
#     saver.restore(sess,ckpt.model_checkpoint_path)
    
#     zs = np.random.uniform(-1.0,1.0,size=[batch_size_sample,z_size]).astype(np.float32) #Generate a random z batch
#     newZ = sess.run(Gz,feed_dict={z_in:z2}) #Use new z to get sample images from generator.
#     if not os.path.exists(sample_directory):
#         os.makedirs(sample_directory)
#     save_images(np.reshape(newZ[0:batch_size_sample],[36,32,32]),[6,6],sample_directory+'/fig'+str(i)+'.png')

