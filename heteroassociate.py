import Tree
import Unit
import MHN
import math
import torch
from torch import nn
import torchvision
import numpy as np
import pickle
import utilities
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data import random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Subset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

softmax = nn.Softmax(dim=1)
NLL = nn.NLLLoss(reduction='sum')
mse = torch.nn.MSELoss(reduction='sum')
bce = torch.nn.BCELoss(reduction='sum')
cos = torch.nn.CosineSimilarity(dim=0)
relu = nn.ReLU()

class PairedMNIST(Dataset):
    def __init__(self, digit_pairs=[(2,9), (3,4), (7,8), (1,5), (6,0)], num_pairs_per_class=10000, root='./data'):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        
        # Load MNIST dataset
        mnist = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)

        self.pairs = []
        self.labels = []

        for digit1, digit2 in digit_pairs:
            # Get images of each digit
            imgs1 = mnist.data[mnist.targets == digit1]
            imgs2 = mnist.data[mnist.targets == digit2]

            # Randomly select images to create fixed number of pairs
            num_pairs = min(num_pairs_per_class, len(imgs1), len(imgs2))
            idx1 = torch.randperm(len(imgs1))[:num_pairs]  # Random indices for digit1
            idx2 = torch.randperm(len(imgs2))[:num_pairs]  # Random indices for digit2

            # Select the images using the random indices
            paired_imgs1 = imgs1[idx1]
            paired_imgs2 = imgs2[idx2]

            # Concatenate all images in a **single operation** (no loops!)
            paired_batch = torch.cat((paired_imgs1, paired_imgs2), dim=2)  # Concatenate along width

            # Store the paired images and labels (as a tuple)
            self.pairs.append(paired_batch)
            self.labels.extend([(digit1, digit2)] * num_pairs)  # Add labels for each pair

        # Stack all pairs into a single tensor
        self.pairs = torch.cat(self.pairs, dim=0)  # Now it's a tensor instead of a list
        print("Paired batch shape:", paired_batch.shape)
        print(f"Total pairs created: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Return the image and the corresponding pair of labels
        image = self.pairs[idx].unsqueeze(0).float() / 255.0  # Normalize & add channel dim
        labels = self.labels[idx]  # Get the corresponding labels
        return image, labels  # Return both image and labels as a tuple

#Reduces data to specified number of examples per category
def get_data(shuf=True, data=2, btch_size=1):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    dataset = PairedMNIST(digit_pairs=[(2, 9), (3, 4), (7, 8), (1, 5), (6, 0)], num_pairs_per_class=10000)

    test_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=btch_size,
        shuffle=shuf,
        drop_last=True  # <-- Drop last partial batch
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=btch_size,
        shuffle=False,
        drop_last=True  # Optional, if you want full-size batches during test
    )

    return train_loader, test_loader



#Plot Images
def plot_ex(imgs, name):
    grid = make_grid(imgs, nrow=5)
    imgGrid = torchvision.transforms.ToPILImage()(grid)
    plt.imshow(imgGrid)
    plt.axis('off')
    plt.title(name)
    plt.show()


#
def create_train_model(num_imgs, mod_type, data, dev='cuda', act=0):
    # Memorize
    train_loader, test_loader = get_data(shuf=True, data=data, btch_size=num_imgs)
    print("num_images: ", num_imgs)
    for batch_idx, (images, y) in enumerate(train_loader):
        if mod_type == 0:
            model = Unit.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], simFunc=0, actFunc=1).to(dev)
        elif mod_type == 1:
            model = Unit.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], simFunc=4, actFunc=act).to(dev)
        elif mod_type == 2:
            model = Tree.Tree(in_dim=images.view(images.size(0), -1).size(1), chnls=num_imgs, actFunc=act, arch=9, b_sim=4).to(dev)
        elif mod_type == 3:
            model = Tree.Tree(in_dim=images.size(0), chnls=num_imgs, actFunc=act, arch=8, b_sim=4).to(dev)
        elif mod_type == 4:
            model = Unit.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], simFunc=5, actFunc=act).to(dev)
        elif mod_type == 5:
            model = MHN.MemUnit(layer_szs=[images.view(images.size(0), -1).size(1), num_imgs], lr=.03, beta=.00001*num_imgs, optim=1).to(dev)

        images = images.to(dev)
        model.load_wts(images.view(images.size(0), -1))

        return model, train_loader


def noise_test(n_seeds, noise, rec_thr, noise_tp=0, hip_sz=500, mod_type=0, data=0):

    rcll_acc = torch.zeros(n_seeds, len(noise)).to('cpu')
    rcll_mse = torch.zeros(n_seeds).to('cpu')
    for s in range(n_seeds):
        mem_unit, mem_images = create_train_model(hip_sz, mod_type, data)
        with torch.no_grad():
            mem_images = mem_images.to('cpu')

            for ns in range(len(noise)):
                if noise_tp == 0:
                    imgn = torch.clamp(mem_images + torch.randn_like(mem_images) * noise[ns], min=0.000001, max=1).to('cuda')
                else:
                    imgn = (mem_images + torch.randn_like(mem_images) * noise[ns]).to('cuda')


                #Recall and free up gpu memory
                p = mem_unit.recall(imgn).to('cpu')

                rcll_acc[s, ns] += ((torch.mean(torch.square((mem_images.reshape(hip_sz, -1) - p.reshape(hip_sz, -1))),
                                                 dim=1) <= rec_thr).sum() / hip_sz)
                rcll_mse[s] += torch.mean(torch.square(mem_images.reshape(hip_sz, -1) - p.reshape(hip_sz, -1)))

                p.to('cpu')
                imgn.to('cpu')
            mem_images.to('cpu')
            mem_unit.to('cpu')


    return torch.mean(rcll_acc, dim=0), torch.std(rcll_acc, dim=0), torch.mean(rcll_mse, dim=0), torch.std(rcll_mse, dim=0)



def pixDrop_test(n_seeds, frc_msk, rec_thr, hip_sz=500, mod_type=0, data=0):

    rcll_acc = torch.zeros(n_seeds, len(frc_msk))
    rcll_mse = torch.zeros(n_seeds)
    for s in range(n_seeds):
        mem_unit, mem_images = create_train_model(hip_sz, mod_type, data)
        with torch.no_grad():
            mem_images = mem_images.to('cpu')

            for msk in range(len(frc_msk)):
                imgDrop = mem_images.clone().to('cuda')
                mask = torch.rand_like(mem_images[0]) > frc_msk[msk]
                mask = mask.repeat(mem_images.size(0),1,1,1).to('cuda')

                for c in range(imgDrop.size(1)):
                    imgDrop[:, c, :, :] *= mask[:,0,:,:]

                p = mem_unit.recall(imgDrop).to('cpu')
                mask = mask.to('cpu')
                p = p.view(p.size(0), 3, mem_images.size(2), mem_images.size(3))
                rcll_acc[s, msk] += ((torch.mean(torch.square(((mem_images - p) * mask).reshape(hip_sz, -1)),
                                    dim=1) <= rec_thr).sum() / hip_sz)
                rcll_mse[s] += torch.mean(torch.square((mem_images - p) * mask).reshape(hip_sz, -1))

                imgDrop.to('cpu')


    return torch.mean(rcll_acc, dim=0), torch.std(rcll_acc, dim=0), torch.mean(rcll_mse, dim=0), torch.std(rcll_mse, dim=0)




def right_mask_test(n_seeds, frc_msk, rec_thr, hip_sz=1568, mod_type=0, data=0):
    rcll_acc = torch.zeros(n_seeds, len(frc_msk))
    rcll_mse = torch.zeros(n_seeds)

    for s in range(n_seeds):
        mem_unit, test_loader = create_train_model(hip_sz, mod_type, data)

        for batch_idx, (test_images, _) in enumerate(test_loader):
            break  # just grab one batch

        test_images = test_images.to('cpu')

        with torch.no_grad():

            for msk in range(len(frc_msk)):
                imgMsk = test_images.clone()
                imgMsk = imgMsk.to('cuda')
                plt.imsave(f"./images/output_image_{s}_{msk}.png", imgMsk[0].cpu().squeeze(), cmap="gray")

                # Masking operation for grayscale (single-channel) images
                imgMsk[:, :, :, int(test_images.size(2) - test_images.size(2) * frc_msk[msk]):] = 0.

                plt.imsave(f"./images/output_image_msk_{s}_{msk}.png", imgMsk[0].cpu().squeeze(), cmap="gray")

                # Perform recall and free up GPU memory
                p = mem_unit.recall(imgMsk).to('cpu')

                # Adjust the reshaping for grayscale images (1 channel)
                p = p.view(p.size(0), 1, test_images.size(2), test_images.size(3))

                plt.imsave(f"./images/output_image_msk_predict_{s}_{msk}.png", p[0].cpu().squeeze(), cmap="gray")

                # Compute metrics
                lng = int(test_images.size(2) * (1 - frc_msk[msk]))
                rcll_acc[s, msk] += ((torch.mean(torch.square(((test_images[:, :, :, 0:lng] - p[:, :, :, 0:lng])).reshape(hip_sz, -1)),
                                                 dim=1) <= rec_thr).sum() / hip_sz)
                rcll_mse[s] += torch.mean(torch.square((test_images[:, :, :, 0:lng] - p[:, :, :, 0:lng])).reshape(hip_sz, -1))

                imgMsk.to('cpu')

    return torch.mean(rcll_acc, dim=0), torch.std(rcll_acc, dim=0), torch.mean(rcll_mse, dim=0), torch.std(rcll_mse, dim=0)









#Test recall
def recall(n_seeds, frcmsk, noise, rec_thr, test_t, hp_sz, mod_type=0, data=0):
    #Pix drop
    if test_t == 0:
        return pixDrop_test(n_seeds, frcmsk, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data)
    #Mask
    elif test_t == 1:
        return right_mask_test(n_seeds, frcmsk, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data)
    #Noise
    elif test_t == 2:
        return noise_test(n_seeds, noise, rec_thr, hip_sz=hp_sz[0], mod_type=mod_type, data=data)




#Trains
def train(model_type=0, num_seeds=10, hip_sz=[1568], frcmsk=[0], noise=[.2], test_t=0, rec_thr=.001, data=2, act=0):
    #Recall
    acc_means, acc_stds, mse_means, mse_stds = recall(num_seeds, frcmsk=frcmsk, noise=noise, rec_thr=rec_thr,
                                                      test_t=test_t, hp_sz=hip_sz, mod_type=model_type, data=data)

    print(f'Data:{data}', f'Test:{test_t}', 'ModType:', model_type, '\nAcc:', acc_means, acc_stds,
          '\nMSE:', mse_means, mse_stds)


    with open(f'data/HeteroA_Model{model_type}_ActF{act}_Test{test_t}_numN{hip_sz}_frcMsk{frcmsk}_data{data}.data', 'wb') as filehandle:
        pickle.dump([acc_means, acc_stds, mse_means, mse_stds], filehandle)