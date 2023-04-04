import numpy as np
import torch
import matplotlib.pyplot as plt
import Pk_library as PKL

fontsiz = 10
cols = ["b","r"]
labels = ["Ground Truth", "DDPM"]

# Power spectrum parameters
BoxSize = 25.0 #h/Mpc
#kmax    = 20.0 #h/Mpc

def sample_plot_image(data, num_images = 5):
    
    fig = plt.figure(figsize=(15,15))
    plt.axis('off')
    
    for j in range(num_images):
        plt.subplot(1, num_images, j+1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(data[j].detach().cpu().squeeze(),cmap="Greys")
        
    plt.subplots_adjust(wspace=0.05)
    plt.show()

# Plot the probability distribution function (pdf), comparing ground truth and predicted images
def plot_pdf(targets,outputs):

    fig, (ax_1) = plt.subplots(1, 1, figsize=(6,6))
    
    listbins = np.linspace(-0.1,1.1,num=100)
    
    for ii, data in enumerate([targets,outputs]):
        
        #data = data - data.mean()
        
        hist, bins = np.histogram(data,density=True,bins=listbins)
        ax_1.plot(bins[:-1],hist,color=cols[ii], alpha=0.5, label=labels[ii])
        
    ax_1.set_ylabel(r"$PDF(\delta)$",fontsize=fontsiz)
    ax_1.set_xlabel(r"$\delta$ (normalized)",fontsize=fontsiz)
    ax_1.set_xlim([bins[0],bins[-1]])
    ax_1.set_yscale("log")
    ax_1.legend(fontsize=fontsiz)
    ax_1.grid()
    #plt.savefig(path+"Plots/PDF"+suf+".pdf")
    #plt.close(fig)

# Plot the power spectrum, comparing ground truth and predicted images
def plot_power_spectrum(arr1, arr2):

    fig = plt.figure(figsize=(6,6))

    for ii, data in enumerate([arr1, arr2]):

        Pks = []

        #data = data - data.mean()

        for i in range(len(data)):

            delta = data[i].squeeze()
            Pk = PKL.Pk_plane(delta.cpu().detach().numpy(), BoxSize, verbose=False)
            k   = Pk.k
            Pk0 = Pk.Pk
            Pks.append(torch.tensor(Pk0).view(1,-1))

        Pks = torch.cat(Pks,dim=0)
        Pk_mean, Pk_std = Pks.mean(0), Pks.std(0)

        plt.plot(k, Pk_mean, color=cols[ii])
        plt.fill_between(k, Pk_mean - Pk_std, Pk_mean + Pk_std, color=cols[ii], alpha=0.5, label=labels[ii])

    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$k$",fontsize=fontsiz)
    plt.ylabel(r"$P(k)$",fontsize=fontsiz)
    plt.xlim([k.min(),k.max()])


