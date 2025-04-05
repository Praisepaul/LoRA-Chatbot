from pdfminer.high_level import extract_text
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import get_peft_model, LoraConfig, TaskType
import torch
from azure.storage.blob import BlobServiceClient
import os

# Azure Blob Storage Configuration
CONNECT_STR = "YOUR_STORAGE_ACCOUNT_CONNECTION_STRING"  # Replace with your connection string
CONTAINER_NAME = "medibot-data"
BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(CONNECT_STR)
CONTAINER_CLIENT = BLOB_SERVICE_CLIENT.get_container_client(CONTAINER_NAME)
MODEL_OUTPUT_DIR = "./lora-medical-chatbot"
BLOB_MODEL_PATH = "trained_model"

def upload_file_to_blob(local_file_path, blob_name):
    try:
        with open(local_file_path, "rb") as data:
            CONTAINER_CLIENT.upload_blob(name=blob_name, data=data, overwrite=True)
        print(f"Uploaded {local_file_path} to {CONTAINER_NAME}/{blob_name}")
    except Exception as e:
        print(f"Error uploading {local_file_path}: {e}")

def download_file_from_blob(blob_name, local_file_path):
    try:
        blob_client = CONTAINER_CLIENT.get_blob_client(blob=blob_name)
        with open(local_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"Downloaded {CONTAINER_NAME}/{blob_name} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading {CONTAINER_NAME}/{blob_name}: {e}")
        return None
    return local_file_path

# Download the medical book from Blob Storage
local_medical_book_path = "medical_book.pdf"
download_file_from_blob("data/medical_book.pdf", local_medical_book_path)

# Extract text from the PDF
try:
    text = extract_text(local_medical_book_path)
except FileNotFoundError:
    print(f"Error: Medical book not found at {local_medical_book_path}. Make sure it's uploaded to Blob Storage.")
    exit()

# Split text into question-answer-like chunks
def split_into_qa_pairs(text, max_chunk_length=256):
    import re
    chunks = []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
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
            for chunk in chunks if len(chunk) > 50]

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
    save_strategy="steps", # Changed to steps for easier saving
    save_steps=500,
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
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return compute_loss(model, inputs, return_outputs)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save the trained model to Blob Storage
local_model_path = "./lora-medical-chatbot"
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)

for filename in os.listdir(local_model_path):
    local_file = os.path.join(local_model_path, filename)
    blob_name = os.path.join(BLOB_MODEL_PATH, filename)
    upload_file_to_blob(local_file, blob_name)

# Load the model from Blob Storage for chatbot (conceptual - you might need to download first)
local_model_for_chat_path = "lora-medical-chatbot-loaded"
os.makedirs(local_model_for_chat_path, exist_ok=True)
download_file_from_blob(os.path.join(BLOB_MODEL_PATH, "adapter_config.json"), os.path.join(local_model_for_chat_path, "adapter_config.json"))
download_file_from_blob(os.path.join(BLOB_MODEL_PATH, "adapter_model.bin"), os.path.join(local_model_for_chat_path, "adapter_model.bin"))
download_file_from_blob(os.path.join(BLOB_MODEL_PATH, "tokenizer_config.json"), os.path.join(local_model_for_chat_path, "tokenizer_config.json"))
download_file_from_blob(os.path.join(BLOB_MODEL_PATH, "special_tokens_map.json"), os.path.join(local_model_for_chat_path, "special_tokens_map.json"))
download_file_from_blob(os.path.join(BLOB_MODEL_PATH, "tokenizer.json"), os.path.join(local_model_for_chat_path, "tokenizer.json"))

chatbot = pipeline("text-generation", model=local_model_for_chat_path, tokenizer=local_model_for_chat_path, device=-1)

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