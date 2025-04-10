from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

model_name = "meta-llama/Llama-3.2-3B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True,return_dict_in_generate=True,torch_dtype = torch.float16,device_map='auto')

text = """[{'role': 'system', 'content': '\nYou are a helpful AI assistant for a coffee shop application.\n          Your task is to determine what agent should handle the user input. You have 3 agents to choose from:\n            1. details_agent: This agent is responsible for answering questions about the coffee shop, like location, delivery places, working hours, details about menue items. Or listing items in the menu items. Or by asking what we have.\n            2. order_taking_agent: This agent is responsible for taking orders from the user. It\'s responsible to have a conversation with the user about the order untill it\'s complete.\n            3. recommendation_agent: This agent is responsible for giving recommendations to the user about what to buy. If the user asks for a recommendation, this agent should be used.\n\n            Your output should be in a structured json format like so. each key is a string and each value is a string. Make sure to follow the format exactly:\n            {\n            "chain of thought": go over each of the agents above and write some your thoughts about what agent is this input relevant to.\n            "decision": "details_agent" or "order_taking_agent" or "recommendation_agent". Pick one of those. and only write the word.\n            "message": leave the message empty.\n            }\n            '}, {'role': 'user', 'content': 'give me full menu'}, {'role': 'assistant', 'content': "\n\nWelcome to Merry's Way! I'm happy to help you with our menu. We have two delicious options for you to choose from today.\n\nFirst, we have our signature Espresso shot, a bold shot of rich espresso made from the finest beans to deliver a robust flavor in every sip. It's perfect for a quick pick-me-up or as a base for your favorite coffee drinks. Our Espresso shot is priced at $2.00 and has a rating of 4.9 out of 5 stars.\n\nNext, we have our Dark Chocolate, a rich and indulgent beverage made with premium cocoa, sugar, and milk. This luxurious treat is perfect for a cozy treat on a chilly day, bringing warmth and comfort with every sip. Our Dark Chocolate is priced at $5.00 and has a rating of 4.7 out of 5 stars.\n\nWould you like to order one of these options, or would you like me to suggest something else?"}, {'role': 'user', 'content': 'order one iced latte for me'}]"""

inputs = tokenizer(text, return_tensors="pt")
inputs.to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, return_dict_in_generate=True, output_hidden_states=True,max_new_tokens = 100)


hidden_states = outputs.hidden_states
last_hidden_state = hidden_states[-1]
response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print("res",response)

# last_hidden_state = last_hidden_state.cpu()
last_hidden_state = list(last_hidden_state)
print("before",type(last_hidden_state),len(last_hidden_state))
last_hidden_state = torch.stack(last_hidden_state,dim=0)

print("after",type(last_hidden_state),last_hidden_state.size())

print(list(last_hidden_state),len(last_hidden_state),type(last_hidden_state))
last_hidden_state = last_hidden_state.cpu()

embedding = np.array(list(last_hidden_state))
print("before",embedding.shape,last_hidden_state.shape)
embedding = embedding.mean(axis=1)
print("after",embedding.shape)

print("Embedding shape:", embedding.shape)
