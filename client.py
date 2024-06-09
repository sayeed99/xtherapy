import os
import json
os.environ['TRANSFORMERS_CACHE'] = '/media/sayeed/projects/huggingface/cache'

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "sayeed99/meta-llama3-8b-xtherapy-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!
formatted_string = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are Anna, a helpful AI assistant for mental therapy assistance developed by a team of developers at xnetics. If you do not know the user's name, start by asking the name. If you do not know details about user, ask them."

# Function to format the string
def format_chat_data(data):
    formatted_output = []
    if data["role"] == "assistant":
        value = data["content"]
        formatted_output.append("<|eot_id|><|start_header_id|>assistant<|end_header_id|>" + value)
    else:
        formatted_output.append("<|eot_id|><|start_header_id|>user<|end_header_id|>" + data["content"])

    return "".join(formatted_output)

def formatting_prompts_funcV2(examples):
    conversations = examples
    text = formatted_string
    for conversation in conversations:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = text + format_chat_data(conversation)
    return text

def get_last_assistant_message(text):
    # Split the text by 'assistant' to isolate assistant's messages
    parts = text.split('<|start_header_id|>assistant<|end_header_id|>')
    
    # The last part is the last assistant message
    # Remove leading/trailing whitespace and return
    last_message = parts[-1].strip()
    last_message = cleanup(last_message)
    return last_message


def cleanup(text):
    # Check if the string ends with 'eot_id'
    if text.endswith('<|eot_id|>'):
        # Remove the last 10 characters
        return text[:-10]
    else:
        return text

# Define a function to handle the conversation and update the session
def handle_conversation(user_input):

    historyPrompt = formatting_prompts_funcV2(user_input)

    historyPrompt = historyPrompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    inputs = tokenizer(
        [
            historyPrompt
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    decoded_outputs = tokenizer.batch_decode(outputs)[0]
    # decoded_outputs = "Hello Welcome"
    last_message = get_last_assistant_message(decoded_outputs)

    # Return the AI response
    return last_message

def complete(messages):
    ai_response = handle_conversation(messages)
    return ai_response