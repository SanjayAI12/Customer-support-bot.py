import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

# Load dataset and label names
dataset = load_dataset("banking77")
label_list = dataset['train'].features['label'].names

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_list))

# Tokenization with fixed padding
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Dataloaders
train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=16)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop (3 epochs)
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch["label"].to(device)

        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f}")

# Inference function
def get_response(user_input):
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(user_input, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
        output = model(**encoded)
        pred = torch.argmax(output.logits, dim=1).item()
        intent = label_list[pred]
        return f"Intent: {intent} — [Placeholder reply for '{intent}']"

# Interactive loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    print("Bot:", get_response(query))
