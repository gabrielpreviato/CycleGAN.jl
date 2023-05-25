using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Statistics
using Zygote
using CUDA

export Discriminator
export Generator

struct Cyclegan
    gen_A
    gen_B
    disc_A
    disc_B
end

@functor Cyclegan


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


function train_discr(cycleGAN, original_A, original_B,
                     fake_A, fake_B, opt_discr)
    ps = Flux.params(cycleGAN.disc_A, cycleGAN.disc_B)
    loss, back = Zygote.pullback(ps) do
                        #calculate A
                        D_A_real = cycleGAN.disc_A(original_A)
                        D_A_fake = cycleGAN.disc_A(fake_A)
                        D_A_real_loss = mse(D_A_real, ones(size(D_A_real)) |> gpu)
                        D_A_fake_loss = mse(D_A_fake, zeros(size(D_A_fake)) |> gpu)
                        D_A_loss = D_A_real_loss + D_A_fake_loss

                        #calculate B
                        D_B_real = cycleGAN.disc_B(original_B)
                        D_B_fake = cycleGAN.disc_B(fake_B)
                        D_B_real_loss = mse(D_B_real, ones(size(D_B_real)) |> gpu)
                        D_B_fake_loss = mse(D_B_fake, zeros(size(D_B_fake)) |> gpu)
                        D_B_loss = D_B_real_loss + D_B_fake_loss
                        calculate_loss_discr(D_A_loss, D_B_loss)
    end
    grads = back(1.f0)
    update!(opt_discr, ps, grads)

    return loss
end

Zygote.@nograd train_discr

function train_gan(cycleGAN, original_A, original_B, opt_gen, opt_discr)
    loss = Dict()
    ps = Flux.params(cycleGAN.gen_A, cycleGAN.gen_B)
    loss["G_loss"], back = Zygote.pullback(ps) do
                            fake_A = cycleGAN.gen_A(original_B)
                            fake_B = cycleGAN.gen_B(original_A)
                            loss["D_loss"]= train_discr(cycleGAN,
                                                        original_A, original_B,
                                                        fake_A, fake_B, opt_discr)
                            # adversarial loss for both generators
                            D_A_fake = cycleGAN.disc_A(fake_A)
                            D_B_fake = cycleGAN.disc_B(fake_B)
                            loss_G_A = mse(D_A_fake, ones(size(D_A_fake)) |> gpu)
                            loss_G_B = mse(D_B_fake, ones(size(D_B_fake)) |> gpu)

                            #cycle loss
                            cycle_B = cycleGAN.gen_B(fake_A)
                            cycle_A = cycleGAN.gen_A(fake_B)
                            cycle_B_loss = mae(original_B, cycle_B)
                            cycle_A_loss = mae(original_A, cycle_A)
                            
                            calculate_loss_gen(loss_G_A, loss_G_B, cycle_A_loss, cycle_B_loss)
    end
    grads = back(1.f0)
    update!(opt_gen, ps, grads)
    return loss
end