import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

def main():
  # 1. Load dataset
  dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

  # 2. Model and tokenizer setup
  model_id = "mistralai/Mistral-7B-v0.1"

  # 4-bit quantization config
  bmb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
  )

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.per_token = tokenizer.eos_token

  # 3. LoRA config
  peft_config = LoraConfig(
    f=10,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
  )

  # 4. Training arguments
  training_args = TrainingArguments(
    output_dir="mistral-7b-qlora-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=8,
    max_steps=100,  # For demo purposes - remove in real training
    fp16=True,
    report_to="none"  # Disable wandb/tensorboard
  )

  #5. Training setup
  trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tenizer=tokenizer,
    packing=True,
  )

  # 6. Train!
  trainer.train()

  # 7. Save the model
  trainer.model.save_pretrained("mistral-7b-qlora-finetuned")

  if __name__ == "__main__":
    main()
