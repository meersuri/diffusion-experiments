import tqdm
import torch
from diffusion.dataset import CIFAR10, transform_Flowers102, Flowers102
from diffusion.models import UNet
import PIL
from PIL import Image
import numpy as np

from diffusion.algorithm import LinearScheduler

def train_diffusion(model, dataset, epochs=1, max_time=1000):
    wts_path = type(dataset).__name__ + '_weights.pth'
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model.to('cuda')
    model.train(True)
    optim = torch.optim.Adagrad(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    best_loss = np.inf
    schedule = LinearScheduler(max_time)
    for epoch in range(epochs):
        for step, (data, label) in enumerate(tqdm.tqdm(loader)):
            x0 = data.to('cuda')
            optim.zero_grad()
            time = torch.randint(0, max_time, (1,)).to('cuda')
            noisy_sample, noise = schedule.sample(x0, time)
            noise_pred = model(noisy_sample, time)
            loss = criterion(noise_pred, noise)
            loss.backward()
            optim.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), wts_path)
            if step % 10 == 0:
                print(f"\033sLoss: {loss.item()}\033u")

def train_autoencoder(model, dataset, epochs=1):
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
            out = model(data, 500)
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

def test_autoencoder(model, dataset):
    wts_path = type(dataset).__name__ + '_weights.pth'
    load(model, wts_path)
    model.eval()
    model.to('cpu')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for data, label in test_loader:
        out = model(data, 0)
        for i in range(out.shape[0]):
            img = out[i, ...]
            img = np.moveaxis(img.detach().numpy(), 0, -1)
            img = 255*np.clip(img, 0.0, 1.0)
            img = img.astype(np.uint8)
            img = PIL.Image.fromarray(img)
            img.save(f'out_{i}.png')
        break

def test_diffusion(model, dataset, shape, steps=None):
    wts_path = type(dataset).__name__ + '_weights.pth'
    load(model, wts_path)
    model.eval()
    model.to('cuda')
    scheduler = LinearScheduler()
    if steps is None:
        steps = scheduler.steps
    data = torch.randn(shape).to('cuda')
    with torch.no_grad():
        for t in tqdm.tqdm(reversed(range(steps))):
            if t == 0:
                z = torch.zeros(shape).to('cuda')
            else:
                z = torch.randn(shape).to('cuda')
            a = scheduler.alphas[t].to('cuda')
            ca = scheduler.cumulative_alphas[t].to('cuda')
            data = (1/torch.sqrt(a))*(data - (1 - a)*model(data, t)/torch.sqrt((1 - ca))) + torch.sqrt(1 - a)*z

            for i in range(data.shape[0]):
                img = data[i, ...].to('cpu')
                img = np.moveaxis(img.detach().numpy(), 0, -1)
                img = 255*np.clip(img, 0.0, 1.0)
                img = img.astype(np.uint8)
                img = PIL.Image.fromarray(img)
                img.save(f'out_{i}.png')

def export_model(model, dataset, time=0):
    model.eval()
    model.to('cpu')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data, label = next(iter(loader))
    torch.onnx.export(model, (data, time), type(dataset).__name__ + '_model.onnx', opset_version=17)

if __name__ == '__main__':
    img_size = (64, 64)
    model = UNet(start_channels=32, factor=8)
    preprocess = transform_Flowers102(img_size=img_size)
    data = Flowers102(train=True, transform=preprocess)
    train_diffusion(model, data, epochs=15)
#    train_autoencoder(model, data, epochs=15)
#    test_autoencoder(model, data, 500)
    test_diffusion(model, data, (8, 3, img_size[0], img_size[0]))
#    export_model(model, data)


