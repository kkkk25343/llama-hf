# llama-hf
This repository contains the implementation for Lab 2 of ID2223 (HT2025), focusing on parameter-efficient fine-tuning (PEFT) of a LLM and deploying inferences UI on Hugging Face Spaces.

## Project Overview
In this lab, we fine-tuned a Llama-3.2 3B model on the FineTome instruction dataset using PEFT techniques to reduce GPU memory usage during training.
The training process was executed in Google Colab (T4 GPU), with periodic checkpoint saving and later conversion to GGUF format for CPU-based inference.

### Key steps implemented
- Set up a reproducible Python environment for LLM fine-tuning  
- Applied **LoRA** for efficient adaptation of the model  
- Trained the model on instruction-following data  
- Exported the model to **GGUF** for lightweight deployment  
- Built a **Gradio UI** and deployed it on Hugging Face Spaces


## Fine-Tuned Models

We experimented with **three different foundation models** to evaluate how model size, architecture, and data variations affect downstream performance and CPU-based inference speed. The fine-tuning processes and results are documented in the corresponding Jupyter notebooks:

1. **Llama_3_2_1B+3B_Conversational_original.ipynb**  
   - Baseline fine-tuning using the original Llama-3.2 3B conversational checkpoints.  
   - Serves as the reference model for later comparisons in both performance and inference latency.

2. **Llama_3_2_3B_Conversational_tuned_params+new_data.ipynb**  
   - Applies a **model-centric approach** by adjusting LoRA hyperparameters and training configuration.  
   - Also applies a **data-centric approach** by extending the instruction dataset with additional samples to improve alignment quality.
     ```python
      dataset1 = load_dataset("mlabonne/FineTome-100k", split="train")
      dataset2 = load_dataset("jondurbin/airoboros-3.2", split="train")
      dataset = concatenate_datasets([dataset1, dataset2])

3. **Phi-3.5-mini-instruct.ipynb**  
   - Fine-tuning an alternative lightweight foundation model to compare with Llama-3 on CPU inference.  
   - Demonstrates that **smaller models can provide faster inference** in the final Gradio UI while maintaining reasonable instruction-following capability.

Together, these experiments fulfill the lab requirement to explore multiple models and evaluate improvements through both model-centric and data-centric strategies.
