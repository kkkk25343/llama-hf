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

2. **Llama_3_2_3B_Conversational_tuned_params+new_data.ipynb**  
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

4. **Phi-3.5-mini-instruct.ipynb**  
   - Fine-tuning an alternative lightweight foundation model to compare with Llama-3 on CPU inference.  
   - Demonstrates that smaller models can provide faster inference in the final Gradio UI while maintaining reasonable instruction-following capability.

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

we create a Hugging Face Space
