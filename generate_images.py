import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
from model import Generator


def show_tensor_images(image_tensor, save_path=None):
    image_tensor = (image_tensor + 1) / 2
    image_grid = torchvision.utils.make_grid(image_tensor)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if save_path:
        plt.savefig(save_path)
    plt.show()

def generate_images(generator, args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    generator.to(device)
    
    noise = torch.randn((args.n_images, args.z_dim, 1, 1)).to(device)

    with torch.no_grad():
        generated = generator(noise).detach().cpu()

    show_tensor_images(generated, args.save_path)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--n_images", type=int, default=32)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--features_g", type=int, default=64)
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()

    assert args.model, "Please pass model checkpoint"

    generator = Generator(args.z_dim, args.image_channels, args.features_g)
    # Load model weights
    checkpoint = torch.load(args.model, weights_only=True)
    generator.load_state_dict(checkpoint['generator_state_dict'])

    generate_images(generator, args)


if __name__ == "__main__":
    main()