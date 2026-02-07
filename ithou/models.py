"""
Model wrapper providing a consistent interface across different model architectures.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Union

from ithou.utils import logger, clear_cuda_cache, log_gpu_memory


class ModelWrapper:
    """Unified interface for interacting with Qwen/Llama language models."""
    
    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name_or_path
        
        logger.info(f"Loading model: {model_name_or_path}")
        log_gpu_memory("Before loading:")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        load_kwargs = {"device_map": device_map, "trust_remote_code": True}
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch_dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        self.model.eval()
        
        log_gpu_memory("After loading:")
        logger.info(f"Model loaded: {self.num_layers} layers, {self.hidden_dim} hidden dim")
    
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
    
    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers
    
    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size
    
    def get_layer(self, layer_idx: int) -> torch.nn.Module:
        """Get a specific transformer layer for hook registration."""
        if hasattr(self.model, "model"):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, "transformer"):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture for {self.model_name}")
    
    def format_messages(
        self,
        system: str,
        user: str,
        assistant: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages using the model's chat template."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})
            add_generation_prompt = False
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    
    def get_prompt_length(self, formatted_prompt: str) -> int:
        """Get the token length of a formatted prompt."""
        return len(self.tokenizer.encode(formatted_prompt, add_special_tokens=False))
    
    def tokenize(self, texts: Union[str, list[str]], padding: bool = True, **kwargs) -> dict:
        """Tokenize text(s) for model input."""
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(texts, padding=padding, return_tensors="pt", **kwargs)
    
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        batch_size: int = 1,
        **kwargs,
    ) -> list[str]:
        """Generate responses for a batch of prompts."""
        all_responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenize(batch_prompts, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            prompt_lengths = [self.get_prompt_length(p) for p in batch_prompts]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    do_sample=do_sample,
                    top_p=top_p if do_sample else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )
            
            for output, prompt_len in zip(outputs, prompt_lengths):
                response_ids = output[prompt_len:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                all_responses.append(response)
            
            del outputs
            clear_cuda_cache()
        
        return all_responses
    
    def extract_activations_batched(
        self,
        prompts: list[str],
        responses: list[str],
        layers: list[int],
        batch_size: int = 8,
    ) -> dict[str, dict[int, list[torch.Tensor]]]:
        """Extract activations at prompt_end, response_start, and response_avg positions."""
        activations = {
            "prompt_end": {layer: [] for layer in layers},
            "response_start": {layer: [] for layer in layers},
            "response_avg": {layer: [] for layer in layers},
        }
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            full_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
            prompt_lengths = [self.get_prompt_length(p) for p in batch_prompts]
            inputs = self.tokenize(full_texts, padding=True)
            
            # Forward pass with hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                    output_hidden_states=True,
                )
            # use detach() to remove the tensor from the computation graph to release GPU memory
            hidden_states = {layer: outputs.hidden_states[layer].detach().cpu() for layer in layers}
            del outputs
            
            # Extract at different positions for each sample
            for j, prompt_len in enumerate(prompt_lengths):
                seq_len = inputs["attention_mask"][j].sum().item()
                response_len = seq_len - prompt_len
                
                if response_len < 1:
                    logger.warning(f"Sample {i+j}: response too short, skipping")
                    continue
                
                for layer in layers:
                    hidden = hidden_states[layer][j]
                    activations["prompt_end"][layer].append(hidden[prompt_len - 1].clone())
                    activations["response_start"][layer].append(hidden[prompt_len].clone())
                    activations["response_avg"][layer].append(hidden[prompt_len:seq_len].mean(dim=0))
            
            del hidden_states
            clear_cuda_cache()
        
        return activations
    
    def __repr__(self) -> str:
        return f"ModelWrapper({self.model_name}, layers={self.num_layers}, hidden_dim={self.hidden_dim})"