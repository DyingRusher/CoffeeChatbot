
from transformers import AutoTokenizer, pipeline,AutoModelForCausalLM
import torch
import numpy as np
import json

def n_get_chatbot_response(pipe,tokenizer,messages,temperature=0):

    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"], "content": message["content"]})
    
    # print("input mes",input_messages,type(input_messages))
    
    prompt = tokenizer.apply_chat_template(
        input_messages, temperature=0.2,
        tokenize=False
    )
    
    out = pipe(prompt,max_new_tokens=1000)

    res = out[0]['generated_text'].split(
                "<|eot_id|>assistant"
            )[1].replace("`","")
    
    # response = double_check_json_output(pipe,tokenizer,response)
    # print("res",res)
    return res

def ne_load_model():
    
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True,return_dict_in_generate=True,torch_dtype = torch.float16,device_map='cuda')

    return pipe,tokenizer,model


def n_get_embedding(model,tokenizer,text_input):

    inputs = tokenizer(text_input, return_tensors="pt")
    inputs.to("cuda")
    print("creating embedding")

    with torch.no_grad():
        outputs = model.generate(**inputs, return_dict_in_generate=True, output_hidden_states=True,max_length = 100)

    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    last = list(last_hidden_state)
    te_la = torch.empty((29,3072))

    for i,te in enumerate(last):
        te_la[i] = te[0][0]
    
    emb = np.array(te_la.cpu()).mean(axis=0)
    
    return emb

def ne_double_check_json_output(model,tokenizer,json_string):


    prompt = f""" You will check this json string and correct any mistakes that will make it invalid.If it is incomplete then complete it and if it is missing any brakets then add it.If there any extra character which will create error then remove it. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.

    Do NOT return a single letter outside of the json string .Do not add explanation of json just return corrected json. Below is the json string:

    {json_string}
    """

    messages = [{"role": "user", "content": prompt}]

    response = n_get_chatbot_response(model,tokenizer,messages)
    print("\ndouble check res",response)

    return response