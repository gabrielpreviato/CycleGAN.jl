using Flux
using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Statistics
using Zygote
using CUDA

using ChainRules: @ignore_derivatives

export Discriminator
export Generator

struct Cyclegan
    gen_A
    gen_B
    disc_A
    disc_B
end

Flux.@functor Cyclegan (gen_A, gen_B, disc_A, disc_B)

function calculate_loss_discr(real, fake)
    return (real + fake) / 2
end

function calculate_loss_gen(loss_G_A, loss_G_B, cycle_A_loss, cycle_B_loss)
    return loss_G_A + loss_G_B  + cycle_A_loss * 10 + cycle_B_loss * 10
end

function train_discr(cycleGAN, opt_disc, original_A, original_B, fake_A, fake_B)
    D_ret = Zygote.withgradient(cycleGAN) do m
        D_A_real = m.disc_A(original_A)
        D_A_fake = m.disc_A(fake_A)
        D_A_real_loss = mse(D_A_real, ones(size(D_A_real)) |> gpu)
        D_A_fake_loss = mse(D_A_fake, zeros(size(D_A_fake)) |> gpu)
        D_A_loss = D_A_real_loss + D_A_fake_loss

        #calculate B
        D_B_real = m.disc_B(original_B)
        D_B_fake = m.disc_B(fake_B)
        D_B_real_loss = mse(D_B_real, ones(size(D_B_real)) |> gpu)
        D_B_fake_loss = mse(D_B_fake, zeros(size(D_B_fake)) |> gpu)
        D_B_loss = D_B_real_loss + D_B_fake_loss
        calculate_loss_discr(D_A_loss, D_B_loss)
    end

    opt_disc, cycleGAN = Flux.update!(opt_disc, cycleGAN, D_ret.grad[1])
end

Zygote.@nograd train_discr

function train_gan(cycleGAN, opt, opt_disc, original_A, original_B)
    G_ret = Zygote.withgradient(cycleGAN) do m
        fake_A = m.gen_A(original_B)
        fake_B = m.gen_B(original_A)

        train_discr(m, opt_disc, original_A, original_B, fake_A, fake_B)

        # adversarial loss for both generators
        D_A_fake = m.disc_A(fake_A)
        D_B_fake = m.disc_B(fake_B)
        loss_G_A = mse(D_A_fake, ones(size(D_A_fake)) |> gpu)
        loss_G_B = mse(D_B_fake, ones(size(D_B_fake)) |> gpu)

        #cycle loss
        cycle_A = m.gen_A(fake_B)
        cycle_B = m.gen_B(fake_A)
        cycle_A_loss = mae(original_A, cycle_A)
        cycle_B_loss = mae(original_B, cycle_B)
        
        calculate_loss_gen(loss_G_A, loss_G_B, cycle_A_loss, cycle_B_loss)
    end

    opt, cycleGAN = Flux.update!(opt, cycleGAN, G_ret.grad[1])
    return G_ret.val[1]
end