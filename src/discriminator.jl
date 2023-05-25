struct Discriminator
    initial
    model
end

@functor Discriminator
function Discriminator(
    in_channels::Int = 3,
    features::Any = [64, 128, 256, 512]
) 
    layers = []
    channel = features[1]
    for index in range(2, length(features))
            if features[index] != last(features)
                push!(layers, 
                    Block(channel, features[index], stride=2)
                )
            else
                push!(layers, 
                    Block(channel, features[index], stride=1)
                )
            end
            channel = features[index]
    end
    push!(layers,
        Conv((4,4), channel => 1; stride=1, pad=1)    
    )
return Discriminator(
    Chain(
        Conv((4,4), in_channels=>features[1]; stride=2, pad=1),
        x -> leakyrelu.(x, 0.2)
    ),
    Chain(layers...)
) |> gpu
end

function (net::Discriminator)(x)
    input = net.initial(x)
    return sigmoid.(net.model(input))
end

using Random
function test()
    img_channels = 3
    img_size = 100
    ## need to explicity type to avoid Slow fallback implementation 
    ## https://discourse.julialang.org/t/flux-con-warning/49456
    x = randn(Float32, (img_size, img_size, img_channels, 5))
    preds = Discriminator()
    println(size(preds(x)))
end