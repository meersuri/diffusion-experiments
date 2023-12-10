import tqdm
import torch
from diffusion.dataset import CIFAR10
from diffusion.models import UNet
import PIL
from PIL import Image
import numpy as np

def train(model):
    data = CIFAR10()
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    model.to('cuda')
    model.train(True)
    optim = torch.optim.Adagrad(model.parameters())
    criterion = torch.nn.MSELoss()
    best_loss = np.inf
    for step, (data, label) in enumerate(tqdm.tqdm(loader)):
        data = data.to('cuda')
        optim.zero_grad()
        out = model(data)
        loss = criterion(out, data)
        loss.backward()
        optim.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'cifar_weights.pth')
        if step % 100 == 0:
            print(f"\033sLoss: {loss.item()}\033u")

def load(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model

def test(model):
    load(model, 'cifar_weights.pth')
    model.eval()
    model.to('cpu')
    test_loader = torch.utils.data.DataLoader(CIFAR10(train=False), batch_size=8, shuffle=True)
    for data, label in test_loader:
        out = model(data)
        for i in range(out.shape[0]):
            img = out[i, ...]
            img = np.moveaxis(img.detach().numpy(), 0, -1)
            img += img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)
            img = PIL.Image.fromarray(img)
            img.save(f'out_{i}.png')
        break

if __name__ == '__main__':
    model = UNet(start_channels=8, factor=4)
    test(model)


