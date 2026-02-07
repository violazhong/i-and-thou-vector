"""
Steering module for applying activation steering during generation.
"""

import torch
import pandas as pd
from typing import Union
from tqdm import tqdm

from ithou.models import ModelWrapper
from ithou.utils import logger, clear_cuda_cache


class SteeringHook:
    """Context manager for applying steering vectors during generation."""
    
    def __init__(
        self,
        model: ModelWrapper,
        vector: torch.Tensor,
        layer: int,
        coefficient: float,
    ):
        """
        Initialize steering hook.
        
        vector: [num_layers, hidden_dim] or [hidden_dim]
        coefficient: positive = toward "I am X", negative = toward "You are X"
        """
        self.model = model
        self.layer = layer
        self.coefficient = coefficient
        self.handle = None
        
        self.vector = vector[layer].clone() if vector.dim() == 2 else vector.clone()
        self.vector = self.vector.to(model.device)
    
    def _hook_fn(self, module, input, output):
        """Forward hook that adds the steering vector."""
        if isinstance(output, tuple):
            hidden = output[0]
            modified = hidden + self.coefficient * self.vector.to(hidden.dtype)
            return (modified,) + output[1:]
        else:
            return output + self.coefficient * self.vector.to(output.dtype)
    
    def __enter__(self):
        layer_module = self.model.get_layer(self.layer)
        self.handle = layer_module.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class MultiLayerSteeringHook:
    """Context manager for applying steering across multiple layers."""
    
    def __init__(
        self,
        model: ModelWrapper,
        vector: torch.Tensor,
        layers: list[int],
        coefficients: Union[float, list[float]],
    ):
        self.model = model
        self.vector = vector
        self.layers = layers
        self.coefficients = [coefficients] * len(layers) if isinstance(coefficients, (int, float)) else coefficients
        assert len(self.coefficients) == len(layers)
        self.hooks = []
    
    def __enter__(self):
        for layer, coef in zip(self.layers, self.coefficients):
            hook = SteeringHook(self.model, self.vector, layer, coef)
            hook.__enter__()
            self.hooks.append(hook)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.__exit__(exc_type, exc_val, exc_tb)
        self.hooks = []


def run_steering_experiment(
    model: ModelWrapper,
    vector: torch.Tensor,
    prompt: str,
    layers: list[int],
    coefficients: list[float],
    max_new_tokens: int,
    system_prompt: str = "You are a helpful assistant."
) -> pd.DataFrame:
    """Run steering experiments across layers and coefficients, return DataFrame."""
    results = []
    formatted_prompt = model.format_messages(system_prompt, prompt)
    total = len(layers) * len(coefficients)
    
    logger.info(f"Running {total} steering experiments")
    
    with tqdm(total=total, desc="Steering experiments") as pbar:
        for layer in layers:
            for coef in coefficients:
                with SteeringHook(model, vector, layer, coef):
                    response = model.generate([formatted_prompt], max_new_tokens=max_new_tokens, do_sample=False)[0]
                
                results.append({"prompt": prompt, "layer": layer, "coefficient": coef, "response": response})
                pbar.update(1)
                clear_cuda_cache()
    
    return pd.DataFrame(results)


def steering_demo(
    model_name: str,
    vector_path: str,
    prompt: str,
    layer: int = 20,
    coefficients: list[float] = [-2, -1, 0, 1, 2],
) -> None:
    """Quick demo of steering effects. Only supports qwen2.5-7b."""
    vector = torch.load(vector_path, map_location="cpu")
    print(f"Loaded vector from {vector_path}, shape: {vector.shape}")
    
    print(f"\nLoading model: {model_name}")
    model = ModelWrapper(model_name)
    
    formatted = model.format_messages("You are a helpful assistant.", prompt)
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Layer: {layer}")
    print(f"{'='*60}\n")
    
    for coef in coefficients:
        direction = "baseline" if coef == 0 else ("I am X" if coef > 0 else "You are X")
        print(f"Coefficient {coef:+.1f} ({direction}):")
        print("-" * 40)
        
        with SteeringHook(model, vector, layer, coef):
            response = model.generate([formatted], max_new_tokens=200, do_sample=False)[0]
        
        print(response[:500])
        if len(response) > 500:
            print("... [truncated]")
        print()