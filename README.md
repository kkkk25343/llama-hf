# llama-hf
This repository contains the implementation for Lab 2 of ID2223 (HT2025), focusing on parameter-efficient fine-tuning (PEFT) of a LLM and deploying inferences UI on Hugging Face Spaces.

## Project Overview
In this lab, we fine-tuned a Llama-3.2 3B model on the FineTome instruction dataset using PEFT techniques to reduce GPU memory usage during training.
The training process was executed in Google Colab (T4 GPU), with periodic checkpoint saving and later conversion to GGUF format for CPU-based inference.

### Model Interfaces

We deployed three separate inference UIs on Hugging Face Spaces to compare model behavior and performance:

- **Llama-3 Conversational (Original Fine-Tuning)**  
  https://stevendhasoi-iris233.hf.space/

- **Llama-3 Conversational (Tuned LoRA Parameters + Extended Dataset)**  
  https://stevendhasoi-newllama.hf.space/

- **Phi-3.5 Mini Instruct (Lightweight Baseline Model)**  
  https://stevendhasoi-phi2223.hf.space/


### Key steps implemented
- Set up a reproducible Python environment for LLM fine-tuning  
- Applied **LoRA** for efficient adaptation of the model  
- Trained the model on instruction-following data  
- Exported the model to **GGUF** for lightweight deployment  
- Built a **Gradio UI** and deployed it on Hugging Face Spaces


## Fine-Tuned Models

We experimented with **three different foundation models** to evaluate how model size, architecture, and data variations affect downstream performance and CPU-based inference speed. The fine-tuning processes and results are documented in the corresponding Jupyter notebooks:

1. **Llama_3_2_3B_Conversational_original.ipynb**  
   - Baseline fine-tuning using the original Llama-3.2 3B conversational checkpoints.  
   - Serves as the reference model for later comparisons in both performance and inference latency.

2. **Llama_3_2_3B_Conversational_tuned_params+new_data.ipynb** (Task **2.1**)
   - Applies a **model-centric approach** by adjusting LoRA hyperparameters and training configuration.
     ```python
     model = FastLanguageModel.get_peft_model(
         model,
         r = 16,              # increased from default 8
         lora_alpha = 32,     # increased scaling factor
         lora_dropout = 0.05, # added dropout for regularization
     )
     ```
     ```python
     training_args = TrainingArguments(
        # warmup_steps: increased from 5 → 10
        warmup_steps = 10,    
        # learning_rate: reduced from 2e-4 → 1e-4 for more stable convergence
        learning_rate = 1e-4
     )
     ```     
   - Also applies a **data-centric approach** by extending the instruction dataset with additional samples to improve alignment quality.
     ```python
      dataset1 = load_dataset("mlabonne/FineTome-100k", split="train")
      dataset2 = load_dataset("jondurbin/airoboros-3.2", split="train")
      dataset = concatenate_datasets([dataset1, dataset2])
     ```

4. **Phi-3.5-mini-instruct.ipynb**  (Task **2.2**)
   - Fine-tuning an alternative lightweight foundation model to compare with Llama-3 on CPU inference.  
   - Experimental results show that the smaller model does not provide noticeable speed improvements, and its instruction-following quality is consistently weaker.

Together, these experiments fulfill the lab requirement to explore multiple models and evaluate improvements through both model-centric and data-centric strategies.

## GGUF Conversion
While attempting to convert the fine-tuned model to **GGUF** directly in Colab using:
```python
model.push_to_hub_gguf(...)
```
we encountered a critical issue: Colab’s RAM usage would spike and eventually crash the runtime.
To resolve this, we first ave the merged LoRA weights locally without converting.
```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
```
After downloading it, we manually ran `llama.cpp` on a local machine with sufficient memory to convert the model to GGUF. 
```python
!python /content/llama.cpp/convert_hf_to_gguf.py \
    /content/drive/MyDrive/llama-3.2-3B/model2 \
    --outfile llama-3.2-3b-f16.gguf \
    --outtype f16
```
After obtaining the `model_f16.gguf` file from the offline GGUF conversion, we further quantized the model using `llama.cpp` to reduce size and improve CPU inference performance.
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake ..
make -j
```
Once compiled, we ran the quantization tool to convert the full-precision GGUF model into a lighter 4-bit variant.
```python
./quantize ../model_f16.gguf ../model_q4_k_m.gguf q4_k_m
```

## Hugging Face Deployment

After obtaining the full-precision GGUF model, we uploaded the quantized checkpoint to Hugging Face so it could be used for inference inside a Space.  
The upload was performed using the following command:

```bash
huggingface-cli upload \
  stevendhasoi/Iriseder \
  /content/model_q4_k_m.gguf \
  --repo-type=model
```

After uploading the model, we created a new Hugging Face Space and replaced the default app.py with the script below:

```python

import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Download your model automatically
model_path = hf_hub_download(
    repo_id="stevendhasoi/Iriseder",
    filename="model_q4_k_m.gguf"
)

# Load GGUF model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
)

def chat_fn(message, history):
    prompt = ""
    for user, bot in history:
        prompt += f"User: {user}\nAssistant: {bot}\n"

    prompt += f"User: {message}\nAssistant:"

    output = llm(
        prompt,
        max_tokens=256,
        stop=["User:"],
        echo=False
    )

    reply = output["choices"][0]["text"].strip()
    return reply

gr.ChatInterface(chat_fn).launch()
```

We then added a requirements.txt file with the necessary dependencies.
Once the Space finished building, the web UI successfully displayed the chat interface using the quantized GGUF Llama model. 



### UI 
This is a demo of how our UI is 
<img width="3072" height="1582" alt="74c491aa4dcfc9a61a171affe7fe7ec9" src="https://github.com/user-attachments/assets/4347db4b-940d-47f7-a242-271067b7c9f5" />

### Result Comparisons
We asked the same two questions on our 3 LLMs. Below are the responses:
<img width="3072" height="1582" alt="e9ac9cd58215819e4a84c9de36c7af52" src="https://github.com/user-attachments/assets/488cba34-3c2a-4d77-b327-e9631f02ce31" />
The original model produced hallucinated facts. For example, it invented members of Roxette (Jonas Berggren is from Ace of Base) and duplicated Queen members. It generated repetitive sentences and incoherent structure. It is a failure response.
<img width="3072" height="1582" alt="791db4981e9b0d61867b5af93d0ab4a3" src="https://github.com/user-attachments/assets/9e7a5613-63a1-4164-8424-40d93c8c3345" />
The parameter-tuned model returned fully correct factual information for both bands, showed more coherent, natural, and structured responses, reduced hallucinations and removed repetition. But hallucination still exists, for example, Roxette did not break up in 1995. The fact is that they even performed in China for the first time in 1995.
The tuned model has the same architecture and parameter size as the original model, so inference speed remains almost unchanged. They are both very slow, with 100s+ to make a response.  
<img width="3072" height="1582" alt="image" src="https://github.com/user-attachments/assets/d9341558-e057-4c4f-bf20-b33e697c407f" />
The Phi-3.5-mini model gives clearer and smoother answers. However, it sometimes talks too much, and occasionally ends a sentence early or gives details that are not fully precise. For example, Queen never won a Grammy award but a lifetime achievement award. They also do not have a song called “Don'aval”.

