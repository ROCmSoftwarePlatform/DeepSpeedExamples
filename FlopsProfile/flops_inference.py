from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.profiling.flops_profiler import FlopsProfiler
import torchvision.models as models
import torch

# Code snippet for Profiling at Inference
# Profiler output to std, and returns the total MACs and parameters of a model.
with torch.cuda.device(0):
    model = models.alexnet()
    batch_size = 256
    # Return: The number of multiply-accumulate operations (MACs) and parameters in the model
    macs, params = get_model_profile(model = model, input_res = (batch_size, 3, 224, 224))

