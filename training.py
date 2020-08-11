import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch import load
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from shutil import rmtree
import time

import agents
from hyperparameters import image_size, batch_size, adam_params_D, adam_params_G, n_epochs, latent_dim, \
    sample_interval, lambda_gp, real_label, fake_label, add_noise, fmsG, fmsD, data_path, load_models, path_g, path_d

start_time = time.time()

# delete the old data
if os.path.isdir('results'):
    print("Delete current results folder? Input y to delete, any other key to exit program")
    keypress = input()
    if keypress == 'y':
        rmtree('results')
    else:
        exit(print("Be sure to change the name of the results folder to keep the content"))
os.makedirs('results')

# load dataset
dataset = dset.ImageFolder(root=data_path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size[1:]),
                               transforms.CenterCrop(image_size[1:]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)
device = torch.device('cuda:0')

# init agents
if image_size[1:] == (4, 4):
    generator = agents.Generator4x4().to(device)
    discriminator = agents.Discriminator4x4().to(device)
elif image_size[1:] == (8, 8):
    generator = agents.Generator8x8().to(device)
    discriminator = agents.Discriminator8x8().to(device)
elif image_size[1:] == (16, 16):
    generator = agents.Generator16x16().to(device)
    discriminator = agents.Discriminator16x16().to(device)
elif image_size[1:] == (32, 32):
    generator = agents.Generator32x32().to(device)
    discriminator = agents.Discriminator32x32().to(device)
elif image_size[1:] == (64, 64):
    generator = agents.Generator64x64().to(device)
    discriminator = agents.Discriminator64x64().to(device)
elif image_size[1:] == (128, 128):
    generator = agents.Generator128x128().to(device)
    discriminator = agents.Discriminator128x128().to(device)

if load_models:
    generator.load_state_dict(load(path_g))
    discriminator.load_state_dict(load(path_d))

# initialize loss function and optimizers
loss = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=adam_params_G[0], betas=adam_params_G[1:])
optimizerD = optim.Adam(discriminator.parameters(), lr=adam_params_D[0], betas=adam_params_D[1:])

# save hyperparameters
file = open('results/hyperparameters.txt', 'w')
file.write('data_path = %s\n' % data_path)
file.write('image_size = (%d, %d, %d)\n' % image_size)
file.write('batch_size = %d\n' % batch_size)
file.write('add_noise = %s\n' % str(add_noise))
file.write('real_label = %f\n' % real_label)
file.write('fake_label = %f\n' % fake_label)
file.write('fmsG = %d\n' % fmsG)
file.write('fmsD = %d\n' % fmsD)
file.write('adam_params_G = (%f, %f, %f)\n' % adam_params_G)
file.write('adam_params_D = (%f, %f, %f)\n' % adam_params_D)
file.write('lambda_gp = %f\n' % lambda_gp)
file.write('latent_dim = %d\n' % latent_dim)
file.write('sample_interval = %d\n' % sample_interval)
file.close()

#################
# Training Loop #
#################
g_losses = []
d_losses = []
g_losses_avg = []
d_losses_avg = []
batches_done = 0
times_saved = 0
for epoch in range(n_epochs):
    print("epoch " + str(epoch))
    for i, data in enumerate(dataloader, 0):
        batches_done += 1
        # Train Discrim

        # reset grads
        discriminator.zero_grad()
        # format real batch
        real = data[0].to(device)
        b_size = real.size(0)
        correct_answer = torch.full((b_size,), real_label, device=device)
        # Pass real batch through
        output = discriminator(real).view(-1)
        # calculate loss and apply gradients
        errD_real = loss(output, correct_answer)
        errD_real.backward()

        # generate fake batch
        z = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake = generator(z)
        correct_answer.fill_(fake_label)
        # Pass fake batch through
        output = discriminator(fake.detach()).view(-1)
        # calculate loss and apply gradients
        grad_pen = agents.compute_gradient_penalty(discriminator, real, fake)
        errD_fake = loss(output, correct_answer)
        errD_fake.backward()
        errD = errD_fake + errD_real + grad_pen * lambda_gp
        d_losses.append(float(errD))
        optimizerD.step()

        # Train Generator

        # reset grads and set correct answer to real (b/c we want the generator to fool the discrim)
        generator.zero_grad()
        correct_answer.fill_(real_label)
        # calculate G's loss based on D's accuracy on the fake samples
        # (rerun fake inputs so G is competing against better D)
        output = discriminator(fake).view(-1)
        errG = loss(output, correct_answer)
        g_losses.append(float(errG))
        errG.backward()
        optimizerG.step()

        if batches_done % sample_interval == 0:
            directory_name = 'results/%depoch%d' % (batches_done, epoch)
            os.makedirs(directory_name)
            print("saving batch", batches_done)
            save_image(fake.data[:25], os.path.join(directory_name, 'images.png'), nrow=5, normalize=True)
            print('saving plot')
            plt.plot(d_losses[sample_interval * times_saved:], label="D", alpha=0.5)
            plt.plot(g_losses[sample_interval * times_saved:], label="G", alpha=0.5)
            g_losses_avg.append(sum(g_losses[sample_interval*times_saved:])/sample_interval)
            d_losses_avg.append(sum(d_losses[sample_interval*times_saved:])/sample_interval)
            plt.plot(sample_interval/2, g_losses_avg[-1], 'r')
            plt.plot(sample_interval/2, d_losses_avg[-1], 'b')
            plt.legend()
            plt.savefig(os.path.join(directory_name, 'lossPlot.png'))
            plt.clf()
            plt.plot(d_losses, label="D", alpha=0.5)
            plt.plot(g_losses, label="G", alpha=0.5)
            d_avg_x = list(range(len(d_losses_avg)))
            d_avg_x = [x * sample_interval for x in d_avg_x]
            g_avg_x = list(range(len(g_losses_avg)))
            g_avg_x = [x * sample_interval for x in g_avg_x]
            plt.plot(d_avg_x, d_losses_avg, label='D_avg', alpha=0.75)
            plt.plot(g_avg_x, g_losses_avg, label='G_avg', alpha=0.75)
            plt.legend()
            plt.savefig('results/TotalLossPlot.png')
            plt.clf()
            print('saving model')
            torch.save(generator.state_dict(), 'results/generator.pt')
            torch.save(discriminator.state_dict(), 'results/discriminator.pt')
            times_saved += 1
            runtime_hours, remainder = divmod(time.time() - start_time, 3600)
            runtime_minutes, runtime_seconds = divmod(remainder, 60)
            print('Model has been training for %d hours, %d minutes, and %f seconds' %
                  (runtime_hours, runtime_minutes, runtime_seconds))
