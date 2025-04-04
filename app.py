
from pdfminer.high_level import extract_text
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Download the medical book from your GitHub repo
# Extract text from the PDF
text = extract_text("medical_book.pdf")

# Split text into question-answer-like chunks
def split_into_qa_pairs(text, max_chunk_length=256): # Reduced chunk length
    import re
    chunks = []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text) # Split on sentence boundaries
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_length:
            current_chunk += sentence.strip() + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence.strip() + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [{"prompt": f"Summarize this:\n{chunk}", "response": chunk}
            for chunk in chunks if len(chunk) > 50]  # Increased min chunk length for better context

qa_data = split_into_qa_pairs(text)

df = pd.DataFrame(qa_data)
df = df.dropna()
dataset = Dataset.from_pandas(df)
dataset = dataset.add_column("text", [f"### Question: {q}\n### Answer: {a}" for q, a in zip(df["prompt"], df["response"])])

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT", torch_dtype=torch.float32)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="no",
    learning_rate=2e-4,
    report_to="none"
)

def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.get("input_ids")
    labels = torch.roll(labels, shifts=-1, dims=1)
    labels[:, -1] = -100
    outputs = model(**inputs, labels=labels)
    return outputs.loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # Add num_items_in_batch
        return compute_loss(model, inputs, return_outputs)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./lora-medical-chatbot")
tokenizer.save_pretrained("./lora-medical-chatbot")

chatbot = pipeline("text-generation", model="./lora-medical-chatbot", tokenizer="./lora-medical-chatbot", device=-1)

# Create a loop for user interaction
def ask_bot():
    print("🩺 Medical Chatbot (type 'exit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("👋 Chat ended.")
            break
        prompt = f"### Question: {user_input}\n### Answer:"
        result = chatbot(prompt, max_new_tokens=150, do_sample=True)[0]["generated_text"]
        answer = result.split("### Answer:")[-1].strip()
        print("Bot:", answer)

ask_bot()