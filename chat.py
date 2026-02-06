from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
# from cpu_trainer import tokenizer
import torch
from peft import PeftModel


base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
checkpoint = "hackathon/my_model_trained"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token 

base_model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                  torch_dtype= torch.float32,
                                                  device_map=None)
                                                  

model = PeftModel.from_pretrained(base_model,checkpoint)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
gen_config = GenerationConfig(do_sample=True,
                              temperature=0.7,
                              top_k=40,
                              eos_token_id=tokenizer.eos_token_id
)

conversation_history = []
print("[+] enjoy.. :)")
print("[+] type 'quit' or 'exit' to stop")

while True:
    user_text = input("Customer: ")
    if user_text.lower() in ["quit", "exit"]:
        print("[+] chat ended... :)")
        break

    # Add user turn
    conversation_history.append(("Customer", user_text))

    # Build prompt with history
    convo_text = "\n".join([f"{speaker}: {text}" for speaker, text in conversation_history])
    prompt = f"""
    ### conversation
    {convo_text}
    Agent:
    """

    # Generate agent response
    outputs = pipe(prompt, generation_config=gen_config)
    agent_reply = outputs[0]["generated_text"].split("Agent:")[-1].strip()

    print("Agent:", agent_reply)

    # Add agent turn
    conversation_history.append(("Agent", agent_reply))