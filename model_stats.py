from model_f0_vqvae import Quantizer
from hparams import get_config
import json
import time
import hparams
import torch
import torch.nn as nn

import torch
from torchinfo import summary

# Assuming you have the Quantizer model class defined and the config 'h'

from easydict import EasyDict as edict

# Create the config dictionary and convert it to an object
h = edict({
    'f0_encoder_params': {
        'input_emb_width': 1,
        'output_emb_width': 128,
        'levels': 1,
        'downs_t': [4],
        'strides_t': [2],
        'width': 32,
        'depth': 4,
        'm_conv': 1.0,
        'dilation_growth_rate': 3
    },
    'f0_vq_params': {
        'l_bins': 20,
        'emb_width': 128,
        'mu': 0.99,
        'levels': 1
    },
    'f0_decoder_params': {
        'input_emb_width': 1,
        'output_emb_width': 128,
        'levels': 1,
        'downs_t': [4],
        'strides_t': [2],
        'width': 32,
        'depth': 4,
        'm_conv': 1.0,
        'dilation_growth_rate': 3
    }
})


# Create the input tensor (x) of shape (batch_size, 80, 112)
x = torch.randn(32, 80, 112)  # Example input with correct shape
x = x[:, 0:1, :]


# Start timing the model creation and summary printing
start_time = time.time()

model = Quantizer(h)

# Print the summary using torchinfo
summary(model, input_data=x)

# End timing
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time for model creation and summary: {elapsed_time:.4f} seconds")

