import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM 
from pydantic import BaseModel
from typing import Union, List, Dict, Optional


class ModelConfig(BaseModel):
    model: str 
    device: str = "auto"
    dtype: str = "bfloat16"
    

class InferenceConfig(BaseModel):
    max_new_tokens: int = 512 
    do_sample: bool = False 
    truncation: bool = True
    temperature: float = 0.7
    skip_special_tokens: bool = True
    

class InferenceEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ) -> None: 
        self.model_config = model_config 
        self.inference_config = inference_config
        self._load_model()
    
    
    def _load_model(self):
        print(f"loading model: {self.model_config.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.model_config.dtype, torch.float16)
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_config.model,
            "torch_dtype": dtype,
            "device_map": self.model_config.device,
        }
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.model.eval()
        
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        print(f"model loaded successfully ({self.n_layers} layers, d_model={self.d_model})")
    
    
    def _format_prompt(
        self,
        prompt: Union[str, List[Dict[str, str]]],
    ) -> str:
        if isinstance(prompt, list):
            prefix_forcing = (
                prompt[-1]["role"].lower() == "assistant" if prompt else False 
            )
            return self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False, 
                add_generation_prompt=not prefix_forcing,
                continue_final_message=prefix_forcing,
            )
        return prompt
    
    
    def _tokenize(self, text: str) -> Dict:
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=self.inference_config.truncation,
        ).to(self.model.device)
    
    
    def infer(
        self,
        prompt: Union[str, List[Dict[str, str]]],
    ) -> List[str]: 
        formatted_prompt = self._format_prompt(prompt)
        inputs = self._tokenize(formatted_prompt)
        prompt_tokens = inputs.input_ids.shape[1]
        
        generate_kwargs = {
            "max_new_tokens": self.inference_config.max_new_tokens,
            "do_sample": self.inference_config.do_sample,
            "temperature": self.inference_config.temperature,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        generated_ids = outputs[:, prompt_tokens:]
        predictions = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=self.inference_config.skip_special_tokens,
        )
        predictions = [pred.strip() for pred in predictions]
        return predictions
    
    
    def get_activations(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        if layers is None:
            layers = list(range(self.n_layers))
        
        activations: Dict[int, torch.Tensor] = {}
        handles = []
        
        for l in layers:
            def _make_hook(idx: int):
                def hook_fn(module, input, output):
                    activations[idx] = output[0].detach().cpu()
                return hook_fn
            h = self.model.model.layers[l].register_forward_hook(_make_hook(l))
            handles.append(h)
        
        formatted_prompt = self._format_prompt(prompt)
        inputs = self._tokenize(formatted_prompt)
        
        with torch.no_grad():
            self.model(**inputs)
        
        for h in handles:
            h.remove()
        
        return activations
    
    
    def get_last_token_activations(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        acts = self.get_activations(prompt, layers)
        return {l: a[0, -1, :] for l, a in acts.items()}
    
    
    def get_activations_batch(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        layers: Optional[List[int]] = None,
    ) -> List[Dict[int, torch.Tensor]]:
        results = []
        for i, prompt in enumerate(prompts):
            acts = self.get_last_token_activations(prompt, layers)
            results.append(acts)
            if (i + 1) % 10 == 0:
                print(f"  processed {i + 1}/{len(prompts)} prompts")
        return results