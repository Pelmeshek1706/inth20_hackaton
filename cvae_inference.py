import torch
from cvae import ConditionalVAE
from torchvision.utils import save_image
from argparse import ArgumentParser
import os
from huggingface_hub import hf_hub_download


def inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    repo_id = "poopaandloopa/int20h_cvae"
    filename = "cvae_epoch_15.pt"
    destination_dir = "weights/"

    parser = ArgumentParser()
    parser.add_argument('--cluster', type=int)
    parser.add_argument('--weights', type=str, required=False, default=None)
    parser.add_argument('--num_classes', type=int, required=False, default=5)
    parser.add_argument('--image_size', type=int, required=False, default=224)
    args = parser.parse_args()

    weights = args.weights

    if not weights:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=destination_dir, local_dir_use_symlinks=False)
        weights = destination_dir + filename

    model_weights = torch.load(weights)['model']

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
