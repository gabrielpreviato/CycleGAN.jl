using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy
using Flux.Losses: mse, mae
using Images
using Statistics
using Parameters: @with_kw
using Random
using Printf
using Zygote

include("generator.jl")
include("discriminator.jl")

size_img = 28

@with_kw struct HyperParams
    batch_size::Int = 32
    epochs::Int = 20
    verbose_freq::Int = 80
    size_dataset::Int = 1000
    lr_dscr_A::Float64 = 0.00005
    lr_gen_A::Float64 = 0.00005
    lr_dscr_B::Float64 = 0.00005
    lr_gen_B::Float64 = 0.00005
end

function convertI2Float(img)
    img_resize = imresize(float.(img), (size_img,size_img))
    if length(size(img_resize)) == 2
        img_resize = RGB.(img_resize)
    end
	return permutedims(channelview(img_resize), (3,2,1))
end

function load_images(path::String, size::Int)
	images=zeros(Float32,size_img,size_img,3,size)
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

function calculate_loss_discr(real, fake)
    return (real + fake) / 2
end

function calculate_loss_gen(loss_G_A, loss_G_B, cycle_A_loss, 
    cycle_B_loss)
    return loss_G_A
        + loss_G_B
        + cycle_A_loss * 10
        + cycle_B_loss * 10
end


function train_discr(discr_A, discr_B, original_A, original_B,
                     fake_A, fake_B, opt_discr)
    ps = Flux.params(discr_A, discr_B)
    loss, back = Zygote.pullback(ps) do
                        #calculate A
                        D_A_real = discr_A(original_A)
                        D_A_fake = discr_A(fake_A)
                        D_A_real_loss = mse(D_A_real, ones(size(D_A_real)))
                        D_A_fake_loss = mse(D_A_fake, zeros(size(D_A_fake)))
                        D_A_loss = D_A_real_loss + D_A_fake_loss

                        #calculate A
                        D_B_real = discr_B(original_B)
                        D_B_fake = discr_B(fake_B)
                        D_B_real_loss = mse(D_B_real, ones(size(D_B_real)))
                        D_B_fake_loss = mse(D_B_fake, zeros(size(D_B_fake)))
                        D_B_loss = D_B_real_loss + D_B_fake_loss
                        calculate_loss_discr(D_A_loss, D_B_loss)
    end
    grads = back(1.f0)
    update!(opt_discr, ps, grads)

    return loss
end

Zygote.@nograd train_discr

function train_gan(gen_A, gen_B, discr_A, discr_B, original_A, original_B, opt_gen, opt_discr)
    loss = Dict()
    ps = Flux.params(gen_A, gen_B)
    loss["G_loss"], back = Zygote.pullback(ps) do
                            fake_A = gen_A(original_B)
                            fake_B = gen_B(original_A)
                            loss["D_loss"]= train_discr(discr_A, discr_B,
                                                        original_A, original_B,
                                                        fake_A, fake_B, opt_discr)
                            # adversarial loss for both generators
                            D_A_fake = discr_A(fake_A)
                            D_B_fake = discr_B(fake_B)
                            loss_G_A = mse(D_A_fake, ones(size(D_A_fake)))
                            loss_G_B = mse(D_B_fake, ones(size(D_B_fake)))

                            #cycle loss
                            cycle_B = gen_B(fake_A)
                            cycle_A = gen_A(fake_B)
                            cycle_B_loss = mae(original_B, cycle_B)
                            cycle_A_loss = mae(original_A, cycle_A)

                            # identity loss (remove these for efficiency if you set lambda_identity=0)
                            #identity_B = gen_B(original_B)
                            #identity_A = gen_A(original_A)
                            #identity_B_loss = mae(original_B, identity_B)
                            #identity_A_loss = mae(original_A, identity_A)
                            
                            calculate_loss_gen(loss_G_A, loss_G_B, cycle_A_loss, cycle_B_loss)
    end
    grads = back(1.f0)
    update!(opt_gen, ps, grads)
    return loss
end

function create_output_image(gen, image)
    @eval Flux.istraining() = false
    fake_image = cpu(gen(image))
    @eval Flux.istraining() = true
    image_array = permutedims(dropdims(fake_image; dims=4), (3,2,1))
    return colorview(RGB, clamp01nan!(image_array))
end

function train()    
    hparams = HyperParams()

    data = load_data(hparams)

    #test images
    test_images_A=zeros(Float32,size_img,size_img,3,1)
    test_images_B=zeros(Float32,size_img,size_img,3,1)
    test_images_A[:,:,:,1] = convertI2Float(load("horse2zebra/testA/n02381460_1000.jpg"))
    test_images_B[:,:,:,1] = convertI2Float(load("horse2zebra/testB/n02391049_100.jpg"))

    # Discriminator
    dscr_A = Discriminator()
    dscr_B = Discriminator() 

    # Generator
    gen_A =  Generator(3, 64) |> gpu
    gen_B =  Generator(3, 64) |> gpu

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr_A, (0.5,0.99))
    opt_gen = ADAM(hparams.lr_gen_A, (0.5,0.99))

    isdir("output")||mkdir("output")
    
    # Training
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for (x,y) in data
                # Update discriminator and generator
            loss = train_gan(gen_A, gen_B, dscr_A, dscr_B, x, y, opt_gen, opt_dscr)
            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(ep), Discriminator loss = $(loss["D_loss"]), Generator loss = $(loss["G_loss"])")
                # Save generated fake image
                output_image_A = create_output_image(gen_A, test_images_B)
                output_image_B = create_output_image(gen_B, test_images_A)
                save(@sprintf("output/cgan_A_steps_%06d.png", train_steps), output_image_A)
                save(@sprintf("output/cgan_B_steps_%06d.png", train_steps), output_image_B)
            end
        train_steps += 1
        end
    end
    println("Finish Training")
    output_image_A = create_output_image(gen_A, test_images_B)
    output_image_B = create_output_image(gen_B, test_images_A)
    save("output/cgan_A_steps_final.png", output_image_A)
    save("output/cgan_B_steps_final.png", output_image_B)
end

cd(@__DIR__)
train()