import glob
import json
import os
import random
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from dotenv import load_dotenv
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()


# --- Custom Local HF Model (Bina Accelerate ke CPU/GPU direct support) ---
class LocalHFModel(DeepEvalBaseLLM):

  def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    print(f"[1/4] Loading tokenizer & model: {model_id}...")
    self.model_id = model_id
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Agar GPU (CUDA) hai toh GPU par load karega, warna CPU par
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
    ).to(self.device)

    # Pad token safe fallback
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    self.pipe = pipeline(
        "text-generation",
        model=self.model,
        tokenizer=self.tokenizer,
    )
    print(f"[1/4] Model loaded successfully on {self.device.upper()}.")

  def load_model(self):
    return self.model

  def generate(self, prompt: str, *args, **kwargs) -> str:
    res = self.pipe(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=self.tokenizer.pad_token_id,
    )
    generated = res[0]["generated_text"]
    if generated.startswith(prompt):
      return generated[len(prompt) :].strip()
    return generated.strip()

  async def a_generate(self, prompt: str, *args, **kwargs) -> str:
    return self.generate(prompt, *args, **kwargs)

  def get_model_name(self):
    return self.model_id


# --- VTT cleaning + chunking ---
def load_chunks():
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  vtt_pattern = os.path.join(base_dir, "data", "*.vtt")
  files = glob.glob(vtt_pattern)

  print(f"[2/4] Searching VTT files at: {vtt_pattern}")
  print(f"[2/4] Found {len(files)} files.")

  if not files:
    raise FileNotFoundError(f"Koi .vtt file nahi mili: {vtt_pattern}")

  texts = []
  for path in files:
    with open(path, encoding="utf-8") as f:
      lines = [
          ln.strip()
          for ln in f
          if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln
      ]
    texts.append(" ".join(lines))

  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
  all_chunks = splitter.split_text("\n\n".join(texts))
  print(f"[2/4] Total chunks created: {len(all_chunks)}")
  return all_chunks


if __name__ == "__main__":
  print("--- Starting Goldens Generation ---")
  chunks = load_chunks()

  # Quick test ke liye 5 sample contexts
  sample = random.sample(chunks, min(5, len(chunks)))
  contexts = [[c] for c in sample]
  print(f"[3/4] Selected {len(contexts)} sample contexts.")

  print("[3/4] Initializing Local Model...")
  local_model = LocalHFModel(model_id="Qwen/Qwen2.5-0.5B-Instruct")
  synthesizer = Synthesizer(model=local_model)

  print("[4/4] Generating goldens...")
  goldens = synthesizer.generate_goldens_from_contexts(
      contexts=contexts,
      include_expected_output=True,
      max_goldens_per_context=1,
  )

  rows = []
  for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",
    })

  os.makedirs("goldens", exist_ok=True)
  out_file = "goldens/retriever_deepeval_goldens.json"
  with open(out_file, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

  print(f"Done! Successfully wrote {len(rows)} goldens to {out_file}")