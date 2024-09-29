import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Critic, Generator, initialize_weights

def gradient_penalty(critic, real, generated, device):
    BATCH_SIZE, C, H, W = real.shape
    eps = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * eps + generated * (1 - eps) # x_hat

    # Calculate critic scores C(x_hat)
    mixed_critic_scores = critic(interpolated_images)

    # Calculate Gradient
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_critic_scores,
        grad_outputs=torch.ones_like(mixed_critic_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    # Flatten Gradient
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1) # L2 Norm
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


def train(generator, critic, dataloader, device, args):
    # Create optimizers
    if args.optimizer == "RMSprop":
        opt_gen = optim.RMSprop(generator.parameters(), lr=args.lr)
        opt_critic = optim.RMSprop(critic.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        opt_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
        opt_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))
    else:
        raise ValueError('Optimezer should be either RMSprop or Adam.')

    fixed_noise = torch.randn((32, args.z_dim, 1, 1)).to(device) # For evaluation

    # Create summary writers to store generated images during evaluation
    writer = SummaryWriter(args.logs_dir + "/" + args.exp_name)
    # Log the arguments
    writer.add_text("Arguments", str(args))
    step=0

    generator.train()
    critic.train

    for epoch in tqdm(range(args.n_epochs)):
        for batch_idx, (real, _) in tqdm(enumerate(dataloader)):
            # Train Critic: max E[C(x)] - E[C(G(z))]
            real = real.to(device)
            # Train critic of critic_iterations iterations
            for _ in range(args.critic_iterations):
                critic_real_output = critic(real).reshape(-1) # Shape: batch_size
                BATCH_SIZE = real.shape[0] #
                noise = torch.randn((BATCH_SIZE, args.z_dim, 1, 1)).to(device)
                generated = generator(noise)
                critic_generated_output = critic(generated).reshape(-1)
                loss_critic = -(torch.mean(critic_real_output) - torch.mean(critic_generated_output))
                # Add the gradient penalty loss
                if args.use_gradient_penalty:
                    gp = gradient_penalty(critic, real, generated, device)
                    loss_critic += args.lambda_gp * gp
                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Enforcing L-Continuity through weight clipping
                if not args.use_gradient_penalty:
                    for p in critic.parameters():
                        p.data.clamp_(-args.weight_clip, args.weight_clip)

            # Train Generator: min -E[C(G(z))]
            noise = torch.randn((args.batch_size, args.z_dim, 1, 1)).to(device)
            critic_generated_output = critic(generator(noise)).reshape(-1)
            loss_gen = -torch.mean(critic_generated_output)
            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Add loss to tensorboard
            writer.add_scalar("generator_loss", loss_gen, epoch * len(dataloader) + batch_idx)
            writer.add_scalar("critic_loss", loss_critic, epoch * len(dataloader) + batch_idx)

            # Print to tensorboard after every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{args.n_epochs}] Batch {batch_idx}/{len(dataloader)} Loss C: {loss_critic:.4f} Loss G: {loss_gen:.4f}")
                with torch.no_grad():
                    generated = generator(fixed_noise)
                    # Get upto 32 samples
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_gen = torchvision.utils.make_grid(generated[:32], normalize=True)

                    writer.add_image("Real", img_grid_real, global_step=step)
                    writer.add_image("Generated", img_grid_gen, global_step=step)

                step += 1

        # Save Model
        torch.save({
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict()
        }, f"{args.save_dir}/{args.exp_name}/epoch_{epoch}.pt")




def main():
    parser = argparse.ArgumentParser("WGAN Training Arguments")
    # DATASET ARGUMENTS
    parser.add_argument("--dataset_name", choices=["MNIST", "FashionMNIST"], help="Dataset Name")
    parser.add_argument("--data_path", default=None, help="Dataset folder path")
    # PARAMETERS / HYPERPARAMETERS
    parser.add_argument("--use_gradient_penalty", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--critic_iterations", type=int, default=5)
    parser.add_argument("--features_d", type=int, default=64)
    parser.add_argument("--features_g", type=int, default=64)
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument("--lambda_gp", type=float, default=10)
    parser.add_argument("--optimizer", choices=["RMSprop", "Adam"], default="RMSprop")
    # OTHERS
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--save_dir", default="models")

    args = parser.parse_args()
    print(args)

    assert args.dataset_name or args.data_path, "Either dataset_name or dataset_path must be provided."

    if args.exp_name is None:
        exp_suffix = "_gp" if args.use_gradient_penalty else ""
        args.exp_name = args.dataset_name if args.dataset_name else args.data_path.replace("/", "_")
        args.exp_name += exp_suffix

    # Create save directory if it does not exist
    os.makedirs(f"{args.save_dir}/{args.exp_name}", exist_ok = True)

    # Create Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(args.image_channels)],
                [0.5 for _ in range(args.image_channels)]
            )
        ]
    )

    # Load Dataset
    if args.dataset_name == "MNIST":
        dataset = datasets.MNIST(root="data/", train=True, transform=transform, download=True)
    elif args.dataset_name == "FashionMNIST":
        dataset = datasets.FashionMNIST(root="data/", train=True, transform=transform, download=True)
    else:
        dataset = datasets.ImageFolder(root=args.data_path, transform=transform)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Models
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    generator = Generator(args.z_dim, args.image_channels, args.features_g).to(device)
    initialize_weights(generator)
    critic = Critic(args.image_channels, args.features_d).to(device)
    initialize_weights(critic)

    train(generator, critic, loader, device, args)


if __name__ == "__main__":
    main()