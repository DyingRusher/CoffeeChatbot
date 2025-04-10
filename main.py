from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
                    OrderTakingAgent,
                    RecommendationAgent,
                    AgentProtocol,
                    
                    ne_load_model
                    )
import os

def main():
    model,tokenizer,emb = ne_load_model()
    
    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()
    recommendation_agent = RecommendationAgent('CoffeeChatbot/recommendation_files/apriori_recommendations.json',
                                                    'CoffeeChatbot/recommendation_files/popularity_recommendation.csv'
                                                    )
    agent_dict: dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "order_taking_agent": OrderTakingAgent(recommendation_agent),
        "recommendation_agent": recommendation_agent
    }
    
    messages = []
    while True:
        # Display the chat history
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Get user input
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response
        guard_agent_response = guard_agent.get_response( model,tokenizer,messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            messages.append(guard_agent_response)
            continue
        
        # Get ClassificationAgent's response
        classification_agent_response = classification_agent.get_response(model,tokenizer,messages)
        chosen_agent=classification_agent_response["memory"]["classification_decision"]
        print("Chosen Agent: ", chosen_agent)

        # Get the chosen agent's response
        agent = agent_dict[chosen_agent]
        if chosen_agent != "details_agent":
            response = agent.get_response(model,tokenizer,messages)
        else:
            response = agent.get_response(model,emb,tokenizer,messages)
        
        messages.append(response)

if __name__ == "__main__":
    main()
