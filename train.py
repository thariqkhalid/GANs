import torch
from torch.autograd.variable import Variable
from torch import optim, nn
from GANs.data_loader import *
from GANs.networks.vanilla import *
from matplotlib import pyplot as plt
from GANs.utils import Logger
from IPython import display
import random


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


def display_data(real_data, fake_data):
    data = [real_data, fake_data]
    r_index = random.sample(range(len(real_data)), 8)
    c = 1

    for r in r_index:
        for d in [0,1]:
            plt.subplot(4, 4, c)
            plt.imshow(torch.reshape(data[d][r], [28,28]).detach().numpy())
            plt.axis('off')
            c = c+1
    plt.show()


def train(discriminator, generator):

    logger = Logger(model_name='VGAN', data_name='MNIST')

    for epoch in range(num_epochs):
        for n_batch, (real_batch, _) in enumerate(data_loader):

            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(real_batch))
            print(len(real_data))
            if torch.cuda.is_available(): real_data = real_data.cuda()
            # Generate fake data
            fake_data = generator(noise(real_data.size(0))).detach()
            print(len(fake_data))

            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                    real_data, fake_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = generator(noise(real_batch.size(0)))
            # Train G
            g_error = train_generator(g_optimizer, fake_data)
            # Log error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Display Progress
            if (n_batch) % 100 == 0:
                display.clear_output(True)
                # Display Images
                test_images = vectors_to_images(generator(test_noise)).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )

                display_data(real_data, fake_data)

            # Model Checkpoints
            logger.save_models(generator, discriminator, epoch)


if __name__ == "__main__":

    # Load data
    data = mnist_data()
    # Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    # Num batches
    num_batches = len(data_loader)

    discriminator = DiscriminatorNet()
    generator = GeneratorNet()
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # Loss function
    loss = nn.BCELoss()

    # Number of steps to apply to the discriminator
    d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
    # Number of epochs
    num_epochs = 200

    num_test_samples = 16
    test_noise = noise(num_test_samples)
    train(discriminator, generator)

