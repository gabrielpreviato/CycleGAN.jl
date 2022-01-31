# CycleGan.js
Julia code of CycleGAN based on pytorch implementation

Code based on implementation from https://github.com/aladdinpersson/Machine-Learning-Collection


### Horses and Zebras Dataset
The dataset can be downloaded from Berkley: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip

### Instalation

```julia
]pkg> add CycleGAN
``` 
or
```julia
julia> import Package; Package.add("CycleGAN")
``` 

### Example of using

```julia
using CycleGAN
using Flux

disc_AB = Discriminator()
disc_BA = Discriminator()

gen_AB = Generator(3, 64)
gen_BA = Generator(3, 64)

cyclegan = Cyclegan(gen_AB, gen_BA, disc_AB,disc_BA)

opt_dscr = ADAM(0.00005, (0.5,0.99))
opt_gen = ADAM(0.00005, (0.5,0.99))

image_A = randn(Float32, (img_size, img_size, img_channels, 1))
image_B = randn(Float32, (img_size, img_size, img_channels, 1))

loss = train_gan(cyclegan, image_A, image_B,opt_gen, opt_dic)
```


## CycleGAN paper
### Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks by Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

#### Abstract
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach. 
```
@misc{zhu2020unpaired,
      title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks}, 
      author={Jun-Yan Zhu and Taesung Park and Phillip Isola and Alexei A. Efros},
      year={2020},
      eprint={1703.10593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
