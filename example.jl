using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Images
using Statistics
using Parameters: @with_kw
using Random
using Printf
using Zygote
using CycleGAN: Discriminator, Generator, Cyclegan, train_gan, train_discr

size_img = 100

@with_kw struct HyperParams
    batch_size::Int = 2
    epochs::Int = 100
    verbose_freq::Int = 50
    size_dataset::Int = 1000
    lr_dscr_A::Float64 = 0.0002
    lr_gen_A::Float64 = 0.0002
    lr_dscr_B::Float64 = 0.0002
    lr_gen_B::Float64 = 0.0002
end

function convertI2Float(img)
    img_resize = imresize(float.(img), (size_img,size_img))
    if length(size(img_resize)) == 2
        img_resize = RGB.(img_resize)
    end
	return permutedims(channelview(img_resize), (3,2,1))
end

function load_images(path::String, size::Int)
	images= zeros(Float32,size_img,size_img,3,size)
	for (index, img) in enumerate(readdir(path, join = true))
		images[:,:,:,index] = convertI2Float(load(img))
        if index == size
            break
        end
	end
	return images
end

function load_data(hparams)
    # Load folder dataset
    images_A = load_images("horse2zebra/trainA/", hparams.size_dataset)
    images_B = load_images("horse2zebra/trainB/", hparams.size_dataset)
    data = [ (images_A[:,:,:, r], images_B[:,:,:, r]) |> gpu for r in partition(1:hparams.size_dataset, hparams.batch_size)]
    return data
end


function create_output_image(gen, image)
    # @eval Flux.istraining() = false
    fake_image = cpu(gen(image))
    # @eval Flux.istraining() = true
    image_array = permutedims(dropdims(fake_image; dims=4), (3,2,1))
    image_array = colorview(RGB, image_array)
    return clamp01nan.(image_array)
end

function train()    
    hparams = HyperParams()

    data = load_data(hparams)

    #test images
    test_images_A=zeros(Float32,size_img,size_img,3,1)
    test_images_B=zeros(Float32,size_img,size_img,3,1)
    test_images_A[:,:,:,1] = convertI2Float(load("horse2zebra/testA/n02381460_1820.jpg"))
    test_images_B[:,:,:,1] = convertI2Float(load("horse2zebra/testB/n02391049_100.jpg"))

    # Discriminator
    disc_A = Discriminator() 
    disc_B = Discriminator()

    # Generator
    gen_A = Generator(3, 64) |> gpu
    gen_B = Generator(3, 64) |> gpu

    # Optimizers
    opt_dscr_a = Flux.setup(ADAM(hparams.lr_dscr_A, (0.5, 0.99)), disc_A)
    opt_dscr_b = Flux.setup(ADAM(hparams.lr_dscr_B, (0.5, 0.99)), disc_B)

    opt_gen_a = Flux.setup(ADAM(hparams.lr_gen_A, (0.5, 0.99)), gen_A)
    opt_gen_b = Flux.setup(ADAM(hparams.lr_gen_B, (0.5, 0.99)), gen_B)

    # opt = (gen_A=opt_gen_a, gen_B=opt_gen_b, disc_A=opt_dscr_a, disc_B=opt_dscr_b)

    cycleGAN = Cyclegan(gen_A, gen_B, disc_A, disc_B)
    opt = Flux.setup(ADAM(hparams.lr_gen_B, (0.5, 0.99)), cycleGAN)

    isdir("output")||mkdir("output")
    
    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        D_loss = 0
        G_loss = 0
        for (i, (x,y)) in enumerate(data)
            l = train_gan(cycleGAN, opt, opt, x, y)
            G_loss += l

            if i % hparams.verbose_freq == 0
                @info("Train step $(i), Discriminator loss = $(D_loss/i), Generator loss = $(G_loss/i)")
                # Save generated fake image
                output_image_A = create_output_image(cycleGAN.gen_A, test_images_B |> gpu)
                output_image_B = create_output_image(cycleGAN.gen_B, test_images_A |> gpu)
                save(@sprintf("output/cgan_A_steps_%06d.png", i), output_image_A)
                save(@sprintf("output/cgan_B_steps_%06d.png", i), output_image_B)
            end
        end
        
    end
    @info("Finish  training")
    output_image_A = create_output_image(cycleGAN.gen_A, test_images_B |> gpu)
    output_image_B = create_output_image(cycleGAN.gen_B, test_images_A |> gpu)
    save("output/cgan_A_steps_final.png", output_image_A)
    save("output/cgan_B_steps_final.png", output_image_B)
end

train()