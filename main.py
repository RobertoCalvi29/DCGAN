import tensorflow as tf
from models import DCGAN

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCHS = 50
    NOISE_DIM = 100
    NUMBER_OF_PREDS = 16
    BUFFER_SIZE = 60000

    # We don't have a test. Unsupervised learning
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # Reshaping from 60k x 28 x 28 into 60k x 28 x 28 x 1 because conv2D needs for each image three dims
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

    # Normalizing between -1 and 1 (that's why we have tanh at the end of the generator)
    train_images = (train_images - 127.5) / 127.5

    # tf dataset from the array
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Defining the model
    GAN = DCGAN(
        image_size=[28, 28, 1],
        latent_dim=NOISE_DIM,
        batch_size=BATCH_SIZE,
        checkpoint_dir='./checkpoints'
    )

    GAN.train(train_dataset, EPOCHS)
