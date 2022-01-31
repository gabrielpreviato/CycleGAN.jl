module CycleGAN

using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Statistics
using Zygote

include("cyclegan.jl")

export Discriminator
export Generator

export Cyclegan

end # end module