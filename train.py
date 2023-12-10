import tqdm
import torch
from diffusion.dataset import CIFAR10, Flowers102
from diffusion.models import UNet
import PIL
from PIL import Image
import numpy as np

def train(model, dataset, epochs=1):
    wts_path = type(dataset).__name__ + '_weights.pth'
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model.to('cuda')
    model.train(True)
    optim = torch.optim.Adagrad(model.parameters())
    criterion = torch.nn.MSELoss()
    best_loss = np.inf
    for epoch in range(epochs):
        for step, (data, label) in enumerate(tqdm.tqdm(loader)):
            data = data.to('cuda')
            optim.zero_grad()
            out = model(data)
            loss = criterion(out, data)
            loss.backward()
            optim.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), wts_path)
            if step % 100 == 0:
                print(f"\033sLoss: {loss.item()}\033u")

def load(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model

def test(model, dataset):
    wts_path = type(dataset).__name__ + '_weights.pth'
    load(model, wts_path)
    model.eval()
    model.to('cpu')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for data, label in test_loader:
        out = model(data)
        for i in range(out.shape[0]):
            img = out[i, ...]
            img = np.moveaxis(img.detach().numpy(), 0, -1)
            img = 255*np.clip(img, 0.0, 1.0)
            img = img.astype(np.uint8)
            img = PIL.Image.fromarray(img)
            img.save(f'out_{i}.png')
        break

def export_model(model, dataset):
    model.eval()
    model.to('cpu')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data, label = next(iter(loader))
    torch.onnx.export(model, data, type(dataset).__name__ + '_model.onnx', opset_version=17)

if __name__ == '__main__':
    model = UNet(start_channels=24, factor=8)
    data = Flowers102(train=True)
#    train(model, data, epochs=10)
    test(model, data)
    export_model(model, data)


