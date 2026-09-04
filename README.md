# pytorch_qlora
## Fine-Tuning Mistral 7B using QLoRA with the peft, transformers and bitsandbytes libraries.

### _Note: This example assumes you're working with a single GPU an A100 or similar_

### Install Prerequisites :pen:
      
```python
      pip install -q bitsandbytes datasets accelerate peft transformers trl
      
```

### Key Notes :spiral_notepad:

* __Quantization__: Uses 4-bit NF4 quantization with double quantization
* __LORA__: Targets all attebtion projection matrices (q/k/v/o_proj)
* __Training__: Uses paged AdamW optimizer with cosine LR schedule.
* __Dataset__: Uses the Guanaco dataset (a cleaned version of OpenAssistant) for instruction tuning.
* __Memory Efficiency__: The combination of QLoRA and packing allows training on a single GPU.

### Build it Docker :hammer_and_wrench:

1.  __Build the Docker image__ :

```bash
      docker build -t mistral-qlora-fintune .
```

2.  __Run the container (with GPU access)__ :
  
```bash
      docker run --gpus all -t --rm \
      -v $(pwd)/output:/workspace/mistral-7b-qlora-finetuned \
      mistral-qlora-finetune
```

* ```--gpus all``` : Gives the container access to all GPUs
* ```-v $(pwd)/output:/workspace/...``` : Mounts a volume to save the trained model

3.  __For production use, you might want to__ 
      
* Remove ```max_steps``` and set proper ```num_train_epochs```
* Add proper volume mounnts for datasets
* Configure logging (eg. TensorBoard or Weights & Biases)

### Hardware Requirements :spiral_notepad:
      
* NVIDIA GPU with at least 16GB VRAM (recommended: A100 40GB)
* Docker with NVIDIA Container Toolkit installed
* At least 30GB of disk space for the image and model

