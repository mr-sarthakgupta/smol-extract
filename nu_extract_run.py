from pydantic import Field, BaseModel
from langchain.llms.base import LLM
from haystack.components.generators import HuggingFaceLocalGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Type
import json
import re
import ast
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

def extract_last_dict(json_string: str) -> dict:
    dict_in_text = ast.literal_eval(json_string.split('### Text')[0])
    return dict_in_text
    
class CarModel(BaseModel):
    Name: str
    Manufacturer: str
    Designers: list
    Number_of_units_produced: int

class DataModel(BaseModel):
    Car: CarModel
car_obj = DataModel(Car=CarModel(Name="", Manufacturer="", Designers=[], Number_of_units_produced=0))

class NuExtract(LLM):
    tokenizer: Any 
    model: Any 
    model_name: str 
    temperature: float
    device: str 
    bound_functions: List[Dict[str, Any]] = []
    function_call: Optional[str] 
    generator: Optional[Any] = None  

    def __init__(self, model_name: str = "numind/NuExtract-1.5", temperature: float = 0.0, device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu', pydantic_example = None, bound_functions: List[Dict[str, Any]] = None, **kwargs):
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer: Any = Field(exclude=True)
        model: Any = Field(exclude=True)
        if bound_functions is None:
            bound_functions = []
        super().__init__(model_name=model_name, temperature=temperature, device=device, tokenizer=tokenizer, model=model, **kwargs)
        self.generator = HuggingFaceLocalGenerator(model="numind/NuExtract-1.5", huggingface_pipeline_kwargs={"model_kwargs": {"torch_dtype":torch.bfloat16}})
        self.generator.warm_up()
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        text = prompt
        pydantic_obj = DataModel(Car=CarModel(Name="", Manufacturer="", Designers=[], Number_of_units_produced=0))
        template = str(pydantic_obj.dict())

        prompt=f"""<|input|>\n### Template:
            {template}           
            ### Text:
            {text}     
            <|output|>
        """
        generated_text = self.generator.run(prompt)      
        extracted_data = extract_last_dict(generated_text['replies'][0])

        print('meow')
        print(text)
        print('meow')

        return extracted_data

    @property
    def _llm_type(self) -> str:
        return "nuextract"

# if __name__ == "__main__":
#     class CarModel(BaseModel):
#         Name: str
#         Manufacturer: str
#         Designers: list
#         Number_of_units_produced: int

#     class DataModel(BaseModel):
#         Car: CarModel
#     car_obj = DataModel(Car=CarModel(Name="", Manufacturer="", Designers=[], Number_of_units_produced=0))

#     text = """
#     The Fiat Panda is a city car manufactured and marketed by Fiat since 1980, currently in its third generation. The first generation Panda, introduced in 1980, was a two-box, three-door hatchback designed by Giorgetto Giugiaro and Aldo Mantovani of Italdesign and was manufactured through 2003 — receiving an all-wheel drive variant in 1983. SEAT of Spain marketed a variation of the first generation Panda under license to Fiat, initially as the Panda and subsequently as the Marbella (1986–1998).

#     The second-generation Panda, launched in 2003 as a 5-door hatchback, was designed by Giuliano Biasio of Bertone, and won the European Car of the Year in 2004. The third-generation Panda debuted at the Frankfurt Motor Show in September 2011, was designed at Fiat Centro Stilo under the direction of Roberto Giolito and remains in production in Italy at Pomigliano d'Arco.[1] The fourth-generation Panda is marketed as Grande Panda, to differentiate it with the third-generation that is sold alongside it. Developed under Stellantis, the Grande Panda is produced in Serbia.

#     In 40 years, Panda production has reached over 7.8 million,[2] of those, approximately 4.5 million were the first generation.[3] In early 2020, its 23-year production was counted as the twenty-ninth most long-lived single generation car in history by Autocar.[4] During its initial design phase, Italdesign referred to the car as il Zero. Fiat later proposed the name Rustica. Ultimately, the Panda was named after Empanda, the Roman goddess and patroness of travelers.
#     """

#     extraction_model = NuExtract()

#     extracted_json = extraction_model.extract_json(car_obj, text)
#     print(extracted_json)