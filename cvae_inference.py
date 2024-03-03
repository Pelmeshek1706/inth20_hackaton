import torch
from cvae import ConditionalVAE
from torchvision.utils import save_image
from argparse import ArgumentParser
import os


def inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--cluster', type=int)
    parser.add_argument('--weights', type=str, required=False, default='weights/cvae_epoch_8.pt')
    parser.add_argument('--num_classes', type=int, required=False, default=7)
    parser.add_argument('--image_size', type=int, required=False, default=224)
    args = parser.parse_args()

    model_weights = torch.load(args.weights)['model']

    model = ConditionalVAE(in_channels=3,
                           num_classes=args.num_classes,
                           latent_dim=128,
                           img_size=args.image_size,
                           device=device)
    model.load_state_dict(model_weights)
    model.eval()
    img_tensor = model.generate(label=args.cluster, device=device)

    if not os.path.exists('inference_results'):
        os.mkdir('inference_results')
    save_image(img_tensor, fp=f'inference_results/cluster_{args.cluster}.jpg')


if __name__ == '__main__':
    inference()

