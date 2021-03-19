# Face-Generator-GANs

In this project, you'll use generative adversarial networks to generate new images of faces.

Get the Data

You'll be using two datasets in this project:

MNIST
CelebA
Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA. Running the GANs on MNIST will allow you to see how well your model trains sooner.

If you're using FloydHub, set data_dir to "/input" and use the FloydHub data ID "R5KrjnANiKVhLWAkpXhNBe".


Explore the Data

MNIST
As you're aware, the MNIST dataset contains images of handwritten digits. You can view the first number of examples by changing show_n_images.

CelebA

The CelebFaces Attributes Dataset (CelebA) dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations. You can view the first number of examples by changing show_n_images.

Preprocess the Data

Since the project's main focus is on building the GANs, we'll preprocess the data for you. The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images. The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.

The MNIST images are black and white images with a single color channel while the CelebA images have 3 color channels (RGB color channel).

Build the Neural Network

You'll build the components necessary to build a GANs by implementing the following functions below:

model_inputs
discriminator
generator
model_loss
model_opt
train

Input

Implement the model_inputs function to create TF Placeholders for the Neural Network. It should create the following placeholders:

Real input images placeholder with rank 4 using image_width, image_height, and image_channels.
Z input placeholder with rank 2 using z_dim.
Learning rate placeholder with rank 0.
Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)

Discriminator

Implement discriminator to create a discriminator neural network that discriminates on images. This function should be able to reuse the variables in the neural network. Use tf.variable_scope with a scope name of "discriminator" to allow the variables to be reused. The function should return a tuple of (tensor output of the discriminator, tensor logits of the discriminator).

Generator

Implement generator to generate an image using z. This function should be able to reuse the variables in the neural network. Use tf.variable_scope with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x out_channel_dim images.

Loss

Implement model_loss to build the GANs for training and calculate the loss. The function should return a tuple of (discriminator loss, generator loss). Use the following functions you implemented:

Optimization

Implement model_opt to create the optimization operations for the GANs. Use tf.trainable_variables to get all the trainable variables. Filter the variables with names that are in the discriminator and generator scope names. The function should return a tuple of (discriminator training operation, generator training operation).

Neural Network Training

Show Output
Use this function to show the current output of the generator during training. It will help you determine how well the GANs is training.

Train

Implement train to build and train the GANs. Use the following functions you implemented:

model_inputs(image_width, image_height, image_channels, z_dim)
model_loss(input_real, input_z, out_channel_dim)
model_opt(d_loss, g_loss, learning_rate, beta1)
Use the show_generator_output to show generator output while you train. Running show_generator_output for every batch will drastically increase training time and increase the size of the notebook. It's recommended to print the generator output every 100 batches.

MNIST

Test your GANs architecture on MNIST. After 2 epochs, the GANs should be able to generate images that look like handwritten digits. Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.
