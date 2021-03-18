import time

import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display
from tensorflow.keras import layers


class DCGAN:
    def __init__(self, image_size, latent_dim, batch_size, checkpoint_dir, img_num=16, save_every=5):
        self.__image_size = image_size  # [height x width x 1] assuming it is black and white
        self.__latent_dim = latent_dim
        self.__batch_size = batch_size
        self.__checkpoint_dir = checkpoint_dir
        self.__img_num = img_num

        self.generator = self.__make_generator_model()
        self.discriminator = self.__make_discriminator_model()

        # Loss for both generator and discriminator
        self.__cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.__generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.__discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.__seed = tf.random.normal([self.__img_num, self.__latent_dim])

        # To save the model
        self.__checkpoint = tf.train.Checkpoint(
            step=tf.Variable(1),
            generator_optimizer=self.__generator_optimizer,
            discriminator_optimizer=self.__discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        self.__manager = tf.train.CheckpointManager(self.__checkpoint, self.__checkpoint_dir, max_to_keep=3)
        self.__save_every = save_every

    def __make_generator_model(self):
        model = tf.keras.Sequential()
        # Dense: transform input vector (latent dim) into 256 low resolution (7x7) images.
        # Note: 7 x 7 works with MNIST (final result is 28 x 28). We don't need bias here
        model.add(layers.Dense(7 * 7 * 256, input_shape=(self.__latent_dim,), use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # This reshapes the output into 256 7x7 "images"
        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
        # Conv2DTranspose is the opposite of convolution. First parameter: how many output images Second parameter:
        # kernel size (height and width of the window). Third parameter: multiplier of the two input dim Padding to
        # pad evenly, so that we don't loose data if the kernel size is not sub-multiple of the size of the input
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # Output here is 64 images of 14x14
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # This will output a single image. Activation is tanh because we normalize the data to be between -1 and 1
        # Instead of 0-255 (black & white image)
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def __make_discriminator_model(self):
        model = tf.keras.Sequential()
        # Output 64 images of shape 14 x 14 assuming input is 28 x 28
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.__image_size))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        # Output 128 images of 7x7
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        # Flattening the output
        model.add(layers.Flatten())
        # Output neuron: one prediction between 0 (fake) and one (real)
        model.add(layers.Dense(1))

        return model

    def __discriminator_loss(self, real_output, fake_output):
        # Real loss: I should predict one for each of the images
        real_loss = self.__cross_entropy(tf.ones_like(real_output), real_output)
        # Fake loss: I should predict 0 for each of the fake images
        fake_loss = self.__cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def __generator_loss(self, fake_output):
        # Gen Loss: the discriminator should predict one for my images
        return self.__cross_entropy(tf.ones_like(fake_output), fake_output)

    def __train_step(self, images):
        # One single training step for both the generator and the discriminator. Will be called by train
        noise = tf.random.normal([self.__batch_size, self.__latent_dim])

        # Standard way to define a training step with gradient tapes
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Will generate BATCH_SIZE images
            generated_images = self.generator(noise, training=True)

            # Feeding the discriminator with both real and fake images
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Computing the two losses
            gen_loss = self.__generator_loss(fake_output)
            disc_loss = self.__discriminator_loss(real_output, fake_output)

            # print(f"Generator loss: {gen_loss}\nDiscriminator loss: {disc_loss}")

        # Computing the two gradients
        gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Backpropagation: optimising the trainable variables according to the two gradients
        self.__generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        self.__discriminator_optimizer.apply_gradients((zip(disc_gradient, self.discriminator.trainable_variables)))

        return gen_loss, disc_loss

    def train(self, data, epochs, from_pretrained=False):
        if from_pretrained:
            self.load_model()

        for epoch in range(epochs):
            start = time.time()
            print("\nStart of epoch %d" % (epoch + 1,))
            for image_batch in data:
                gen_loss, disc_loss = self.__train_step(image_batch)

            display.clear_output(wait=True)
            self.__generate_and_save_images(epoch + 1, self.__seed)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            self.__checkpoint.step.assign_add(1)
            if int(self.__checkpoint.step) % self.__save_every == 0:
                self.__save_model(gen_loss, disc_loss)

        display.clear_output(wait=True)
        self.__generate_and_save_images(epoch + 1, self.__seed)

    def __generate_and_save_images(self, epoch, test_input):
        predictions = self.generator(test_input, training=False)

        plt.figure(figsize=(16, 16))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def __save_model(self, gen_loss, disc_loss):
        save_path = self.__manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.__checkpoint.step), save_path))
        print("Generator Loss {:1.2f}".format(gen_loss.numpy()))
        print("Discriminator Loss {:1.2f}".format(disc_loss.numpy()))

    def load_model(self):
        self.__checkpoint.restore(self.__manager.latest_checkpoint)
        if self.__manager.latest_checkpoint:
            print("Restored from {}".format(self.__manager.latest_checkpoint))
        else:
            print("ERROR: Initializing from scratch.")
