from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LocalHFModel(DeepEvalBaseLLM):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        res = self.pipe(prompt, max_new_tokens=256, do_sample=False)
        return res[0]["generated_text"][len(prompt):].strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_id

# Usage:
# synthesizer = Synthesizer(model=LocalHFModel())