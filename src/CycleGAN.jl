module CycleGAN

using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Statistics
using Zygote

include("network.jl")
include("cyclegan_src.jl")
include("generator.jl")
include("discriminator.jl")

export Discriminator
export Generator

export Cyclegan

export train_gan

end # end module
