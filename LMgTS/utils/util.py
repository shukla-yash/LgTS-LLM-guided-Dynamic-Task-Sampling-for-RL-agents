import torch
import numpy as np
import imageio


def from_numpy(array: np.ndarray, device='cpu'):
    return torch.from_numpy(array).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def save_video(video_record, file='episode0.gif'):
    video_record = [frame.astype(np.uint8) if frame.dtype != np.uint8 else frame for frame in video_record]
    imageio.mimsave('../videos/{}'.format(file), video_record, duration=0.5)
