import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from vae_data import VAEFacesDataset
from torch.utils.tensorboard import SummaryWriter
from utils import one_hot_labels
from torchvision.transforms import Compose, ToTensor, Resize
from cvae import ConditionalVAE
import os
from argparse import ArgumentParser


def train(model, optimizer, n_epochs, n_classes, train_loader):
    if not os.path.exists('weights'):
        os.mkdir('weights')

    writer = SummaryWriter()

    for epoch in range(n_epochs):
        print(f"Epoch: #{epoch}")
        model.train()
        for batch_n, (images, labels) in enumerate(train_loader):
            labels = one_hot_labels(labels, n_classes)

            reconstructed_images, mu, log_var = model(images, labels)
            loss = model.loss_function(reconstructed_images, images, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Total loss', loss.item(), epoch*len(train_loader) + batch_n)

        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   f=f'weights/cvae_epoch_{epoch}.pt')

        model.eval()
        for label in range(n_classes):
            generated_image = model.generate(label).squeeze(dim=0)
            writer.add_image(tag=f'Average image_{label}',
                             img_tensor=generated_image,
                             global_step=epoch)

    writer.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--labels_path', type=str)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--image_size', type=int, required=False, default=224)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--num_epochs', type=int, required=False, default=50)
    args = parser.parse_args()

    transforms = Compose([ToTensor(), Resize((args.image_size, args.image_size))])

    train_dataset = VAEFacesDataset(image_path=args.image_path,
                                    labels_path=args.labels_path,
                                    transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = ConditionalVAE(in_channels=3,
                           num_classes=args.num_classes,
                           latent_dim=128,
                           img_size=args.image_size)
    optimizer = Adam(model.parameters(), lr=3e-4)

    train(model=model,
          optimizer=optimizer,
          n_epochs=args.num_epochs,
          n_classes=args.num_classes,
          train_loader=train_loader)


if __name__ == '__main__':
    main()
