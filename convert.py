"""
This file allows for the convertion of a pytorch model to CoreML
"""

import torch
from torchsummary import summary
import numpy as np

from NNetWrapper import NNetWrapper as NNet
from ReversiGame import ReversiGame

# Load the PyTorch model
game = ReversiGame(8)
model = NNet(game)
model.load_checkpoint('./checkpoints/', 'best.pth.tar')

print(model.nnet)
summary(model.nnet, input_size=(1, 8, 8))

model.nnet.eval()

policy, value = model.predict(game.getInitBoard())

print(policy)
print(value)

# Trace the model and save it to a file
example_input = torch.rand(1, 1, 8, 8)
traced_model = torch.jit.trace(model.nnet, example_input)


import coremltools as ct

# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)]
 )

# Save the converted model.
model.save("AlphaZeroModel.mlpackage")


