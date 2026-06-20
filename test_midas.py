import torch

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

print("MiDaS loaded successfully")