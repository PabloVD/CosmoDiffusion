# CosmoDiffusion

Denoising Diffusion Probabilistic Model (DDPM) in Pytorch to generate CAMELS astrophysical maps.

Adapted from the tutorial [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY) by DeepFindr.

Download images from the [CAMELS Multifield dataset](https://camels-multifield-dataset.readthedocs.io/en/latest/access.html).

In this example we make use of 15k maps of the total mass field at $z=0$ from the LH SIMBA dataset: `Maps_Mtot_SIMBA_LH_z=0.00.npy`

![Sampled images from diffusion model](camels_diffusion.png)
