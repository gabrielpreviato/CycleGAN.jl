struct Generator
    initial
    downblocks
    resblocks
    upblocks
    final
end

@functor Generator
function Generator(
    in_channels::Int,
    num_features::Int = 64,
    num_residual::Int = 9,
)
return Generator(
    Chain(
        Conv((7,7), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    ),
    [
        ConvBlock(3, num_features, num_features*2, true, true;stride=2 ,pad=1), 
        ConvBlock(3, num_features*2, num_features*4, true, true;stride=2 ,pad=1),  
    ],
    Chain([ResidualBlock(num_features*4) for _ in range(1, length=num_residual)]...),
    [
        ConvBlock(3, num_features*4, num_features*2,true,false; stride=2 ,pad=SamePad()), 
        ConvBlock(3, num_features*2, num_features,true,false; stride=2 ,pad=SamePad()),  
    ],
    Conv((7,7),num_features=>in_channels; stride=1,pad=3)
)  |> gpu
end

function (net::Generator)(x)
    input = net.initial(x)
    for layer in net.downblocks
        input = layer(input)
    end
    input = net.resblocks(input)
    for layer in net.upblocks
        input = layer(input)
    end
    return tanh.(net.final(input))
end

using Random
function test()
    img_channels = 3
    img_size = 180
    ## need to explicity type to avoid Slow fallback implementation 
    ## https://discourse.julialang.org/t/flux-con-warning/49456
    x = randn(Float32, (img_size, img_size, img_channels, 2))
    gen = Generator(img_channels, 9)
    println(size(gen(x)))
end

#test()
