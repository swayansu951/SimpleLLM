import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel
from difflib import SequenceMatcher

# Load dataset
with open("Conversational_Transcript_Dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

training_examples = []
for entry in data["transcripts"]:
    context = f"domain: {entry['domain']}\n Intent: {entry['intent']}\n Reason: {entry['reason_for_call']}"
    convo_text = "\n".join([f"{m['speaker']}: {m['text']}" for m in entry["conversation"]])
    training_examples.append({"text": f"### Content\n{context}\n ### Conversation \n {convo_text}"})

dataset = Dataset.from_list(training_examples)

# Load model + tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map=None  # CPU only
)

# LoRA config
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(base_model, lora_config)

# Training args
args = SFTConfig(
    output_dir="hackathon/outputs",
    per_device_train_batch_size=1,
    dataset_text_field="text",
    learning_rate=2e-5,
    max_length=128,
    num_train_epochs=1,
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    logging_steps=1,
)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    processing_class=tokenizer
)
trainer.train()

# Save adapters
model.save_pretrained("hackathon/my_model_trained")

# Reload adapters for inference
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map=None)
model = PeftModel.from_pretrained(base_model, "hackathon/my_model_trained")

# Pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu", torch_dtype=torch.float32)

prompt = """
domain: "E-commerce & Retail"
intent: "Delivery Investigation"
reason_for_call: "Customer James Bailey reported a smart watch showing as delivered but never received, requiring delivery investigation and replacement shipment.",
### conversation
Agent:
"""

outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, eos_token_id=pipe.tokenizer.eos_token_id)
print(outputs[0]["generated_text"])

conversation_history = []

def build_prompt(domain, intent, reason, history):
    convo_text = "\n".join([f"{speaker}: {text}" for speaker, text in history])
    return f"""
    domain: "{domain}"
    intent: "{intent}"
    reason_for_call: "{reason}"
    ### conversation
    {convo_text}
    Agent:
    """


# Build a lookup list from JSON
kb = []
for entry in data["transcripts"]:
    kb.append({
        "domain": entry["domain"],
        "intent": entry["intent"],
        "reason": entry["reason_for_call"],
        "conversation": " ".join([m["text"] for m in entry["conversation"]])
    })

# Simple similarity function
def find_best_match(user_text):
    best_score = 0
    best_entry = None
    for entry in kb:
        score = SequenceMatcher(None, user_text.lower(), entry["conversation"].lower()).ratio()
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry

# Interactive loop
conversation_history = []

while True:
    user_text = input("Customer: ")
    if user_text.lower() in ["quit", "exit"]:
        break

    # Find closest transcript metadata
    match = find_best_match(user_text)
    if match:
        domain, intent, reason = match["domain"], match["intent"], match["reason"]
    else:
        domain, intent, reason = "General", "Unknown", "No reason found"

    # Add user turn
    conversation_history.append(("Customer", user_text))

    # Build prompt
    convo_text = "\n".join([f"{speaker}: {text}" for speaker, text in conversation_history])
    prompt = f"""
    domain: "{domain}"
    intent: "{intent}"
    reason_for_call: "{reason}"
    ### conversation
    {convo_text}
    Agent:
    """

    # Generate agent response
    outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    agent_reply = outputs[0]["generated_text"].split("Agent:")[-1].strip()

    print("Agent:", agent_reply)

    # Add agent turn
    conversation_history.append(("Agent", agent_reply))