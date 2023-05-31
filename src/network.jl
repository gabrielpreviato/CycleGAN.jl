using Flux
using Functors

struct ConvBlock
    conv::Chain
end

@functor ConvBlock
function ConvBlock(
    kernel_size::Int,
    in_channels::Int,
    out_channels::Int,
    act::Bool = true,
    down::Bool = true;
    kwargs...
) 
return ConvBlock(
    Chain(
        if down
            Conv((kernel_size, kernel_size), in_channels => out_channels; kwargs...)
        else
            ConvTranspose((kernel_size, kernel_size), in_channels => out_channels; kwargs...)
        end,
        InstanceNorm(out_channels),
        if act 
           x -> relu.(x) 
        else 
            identity
        end)
)
end

function (net::ConvBlock)(x)
    return net.conv(x)
end

struct ResidualBlock
    block
end

@functor ResidualBlock
function ResidualBlock(
    in_channels::Int
) return ResidualBlock(
    Chain(
        ConvBlock(3, in_channels, in_channels,true,true; pad=1),
        ConvBlock(3, in_channels, in_channels,false,true;pad=1)
    )
)
end

function(net::ResidualBlock)(x)
    return x + net.block(x)
end

struct Block
    conv
end

@functor Block
function Block(
    in_channels::Int,
    out_channels::Int;
    stride::Int
) return Block(
    Chain(
        Conv((4,4), in_channels=>out_channels; stride=stride, pad=1),
        InstanceNorm(out_channels),
        x -> leakyrelu.(x, 0.2)
    )
)
end

function(net::Block)(x)
    return net.conv(x)
end