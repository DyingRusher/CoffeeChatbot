from dotenv import load_dotenv
import os
import json
from copy import deepcopy
from .ne_utils import n_get_chatbot_response,ne_double_check_json_output
load_dotenv()

class GuardAgent():
    def __init__(self):
        pass
    
    def get_response(self,model,tokenizer,messages):
        messages = deepcopy(messages)

        system_prompt = """
            You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
            Your task is to determine whether the user is asking something relevant to the coffee shop or not.
            The user is allowed to:
            1. Ask questions about the coffee shop, like location, working hours, menue items and coffee shop related questions.
            2. Ask questions about menue items, they can ask for ingredients in an item and more details about the item.
            3. Make an order.
            4. ASk about recommendations of what to buy.

            The user is NOT allowed to:
            1. Ask questions about anything else other than our coffee shop.
            2. Ask questions about the staff or how to make a certain menue item.

            Output ONLY the following JSON format — no explanations, no extra text or content.just the JSON no more word:

            {
            "chain of thought": "<explanation>",
            "decision": "<allowed or not allowed>",
            "message": "<leave empty if allowed, else say: 'Sorry, I can't help with that. Can I help you with your order?'>"
            }
            """
        
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        input_messages2 = system_prompt
        chatbot_output = n_get_chatbot_response(model,tokenizer,input_messages)

        # print("res Guard",chatbot_output,"\n")

        chatbot_output = ne_double_check_json_output(model,tokenizer,chatbot_output)
        output = self.postprocess(chatbot_output)
        
        return output

    def postprocess(self,output):

        output = json.loads(output)

        dict_output = {
            "role": "assistant",
            "content": output['message'],
            "memory": {"agent":"guard_agent",
                       "guard_decision": output['decision']
                      }
        }
        return dict_output



    
