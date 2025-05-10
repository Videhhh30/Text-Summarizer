# Text-Summarizer

Here's a clean and informative `README.md` for your PEGASUS fine-tuning project on the SAMSum dataset:

---

# 📚 PEGASUS Fine-Tuning on SAMSum Dataset

This project demonstrates how to fine-tune the [PEGASUS](https://huggingface.co/google/pegasus-cnn_dailymail) transformer model for abstractive summarization on the SAMSum dataset using Hugging Face's `transformers` and `datasets` libraries.

## 🚀 Overview

* Model: `google/pegasus-cnn_dailymail`
* Dataset: [SAMSum](https://huggingface.co/datasets/samsum)
* Task: Abstractive dialogue summarization
* Frameworks: 🤗 Transformers, Datasets, PyTorch
* Evaluation Metric: ROUGE (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)

## 📦 Requirements

Install dependencies:

```bash
pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr
pip install --upgrade accelerate
```

## 🛠️ Training

To fine-tune PEGASUS on the SAMSum dataset:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="pegasus-samsum",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_samsum_pt["train"],
    eval_dataset=dataset_samsum_pt["validation"],
    tokenizer=tokenizer,
    data_collator=collator
)

trainer.train()
```

## 📊 Evaluation

Evaluate the model using ROUGE metrics:

```python
from datasets import load_metric

rouge = load_metric("rouge")
# Run the calculate_metric_on_test_ds() function
```

## 💾 Save and Load

To save and reload your model:

```python
model.save_pretrained("pegasus-samsum-model")
tokenizer.save_pretrained("tokenizer")

# Load later
from transformers import pipeline
pipe = pipeline("summarization", model="pegasus-samsum-model", tokenizer="tokenizer")
```

## 🔍 Sample Prediction

```python
dialogue = dataset_samsum["test"][0]["dialogue"]
summary = pipe(dialogue, num_beams=8, max_length=128, length_penalty=0.8)[0]['summary_text']

print("Dialogue:\n", dialogue)
print("\nGenerated Summary:\n", summary)
```

## 📁 File Structure

```
pegasus-samsum/
│
├── tokenizer/                  # Saved tokenizer
├── pegasus-samsum-model/      # Saved fine-tuned model
├── training_script.py         # Main training code
└── README.md                  # This file
```

## 🧠 Citation

If you use this work, consider citing:

* PEGASUS: [https://arxiv.org/abs/1912.08777](https://arxiv.org/abs/1912.08777)
* SAMSum dataset: [https://arxiv.org/abs/1911.12237](https://arxiv.org/abs/1911.12237)

---

Let me know if you'd like me to auto-generate this into a `.md` file for upload.
