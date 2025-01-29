import getpass
import os
import uuid

uid = uuid.uuid4().hex[:4]  # Avoid conflicts in project names

# Get your API key from https://smith.langchain.com/settings
api_keys = [
    "LANGCHAIN_API_KEY"
]
for key in api_keys:
    if key not in os.environ:
        os.environ[key] = "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"


from langchain_benchmarks import clone_public_dataset, registry

task = registry["Chat Extraction"]

# Clone the dataset to your tenant
clone_public_dataset(task.dataset_id, dataset_name=task.name)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, Dict, Any, List
import json
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, BaseModel
from nu_extract_run import NuExtract

class LocalChatModel(LLM):
    tokenizer: Any = Field(exclude=True)
    model: Any = Field(exclude=True)
    model_name: str = Field(default="numind/NuExtract-1.5")
    temperature: float = Field(default=0.0)
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    bound_functions: List[Dict[str, Any]] = Field(default_factory=list)
    function_call: Optional[str] = Field(default=None)
    nuextract = NuExtract()
    
    def __init__(self, model_name: str = "numind/NuExtract-1.5", temperature: float = 0.0, device: str = None, **kwargs):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model: Any = Field(exclude=True)
        # self.nuextract = NuExtract()
        # self.pydantic_example = pydantic_example
        super().__init__(model_name=model_name, temperature=temperature, device=device, tokenizer=tokenizer, model=model, **kwargs)

    def bind_functions(self, functions: List[Dict[str, Any]], function_call: Optional[str] = None) -> LLM:
        new_instance = self.copy()
        new_instance.bound_functions = functions
        new_instance.function_call = function_call
        return new_instance

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        prompt = f"{prompt}\nOutput only valid JSON"
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        pydantic_example = DataModel(Car=CarModel(Name="", Manufacturer="", Designers=[], Number_of_units_produced=0))

        with torch.no_grad():
            output = self.nuextract.extract_json(pydantic_example, prompt)
        
        print(output)
        print('meow1')
        exit()
            
        # return "{}"

    @property
    def _llm_type(self) -> str:
        return "nuextract"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature, "device": self.device}


class CarModel(BaseModel):
    Name: str
    Manufacturer: str
    Designers: list
    Number_of_units_produced: int

class DataModel(BaseModel):
    Car: CarModel
car_obj = DataModel(Car=CarModel(Name="", Manufacturer="", Designers=[], Number_of_units_produced=0))

model = LocalChatModel()

model("this si prompt")

print('meow')