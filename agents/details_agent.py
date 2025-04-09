from dotenv import load_dotenv
import os
from .ne_utils import n_get_chatbot_response,ne_double_check_json_output,n_get_embedding
from copy import deepcopy
from pinecone import Pinecone
import time
load_dotenv()

class DetailsAgent():
    def __init__(self):
        
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
    
    def get_closest_results(self,index_name,input_embeddings,top_k=2):
        while not self.pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        index = self.pc.Index(index_name)
        em = [0]*3072
        for i in range(3072):
            em[i] = int(input_embeddings[i])

        # print("em",input_embeddings,type(input_embeddings))
        input_embeddings = list(input_embeddings)
        # print("em2",input_embeddings,type((input_embeddings)))
        results = index.query(
            namespace="ns1",
            vector=em,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

        return results

    def get_response(self,pipe,model,tokenizer,messages):
        messages = deepcopy(messages)

        user_message = messages[-1]['content']
        embedding = n_get_embedding(model,tokenizer,user_message)
        result = self.get_closest_results(self.index_name,embedding)

        print("emb",result)
        source_knowledge = "\n".join([x['metadata']['text'].strip()+'\n' for x in result['matches'] ])

        prompt = f"""
        Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {user_message}
        """

        system_prompt = """ You are a customer support agent for a coffee shop called Merry's way. You should answer every question as if you are waiter and provide the neccessary information to the user regarding their orders """
        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output = n_get_chatbot_response(pipe,tokenizer,input_messages)
        # chatbot_output = n_get_chatbot_response(pipe,tokenizer,input_messages)
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self,output):
        output = {
            "role": "assistant",
            "content": output,
            "memory": {"agent":"details_agent"
                      }
        }
        return output

    
