import os
from openai import OpenAI

# Define the model to use
model = "gpt-4o-mini"

# Define the client using your api_key
client = OpenAI(api_key="api_key")

# Define three questions from tourists
tourist_questions = ["How far away is the Louvre from the Eiffel Tower (in miles) if you are driving?","Where is the Arc de Triomphe?","What are the must-see artworks at the Louvre Museum?"]

# Define the conversation to OpenAI client
conversation=[{"role":"system","content":"You are a Parisian expert, delivering valuable insights into the city's iconic landmarks and hidden treasures."}]

for q in tourist_questions:
    print(q)
    
    conversation.append({"role":"user","content":q})

    # Send the messages with temperature and maximum output tokens to the client
    # temperature 0.0 means lowest randomness response
    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        temperature=0.0,
        max_completion_tokens=100
    )

    # Extract the response and append to conversation as an "assistant"
    conversation.append({"role":"assistant","content":response.choices[0].message.content})

    # Send the new messages again 
    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        temperature=0.0,
        max_completion_tokens=100
    )
    
    print(response.choices[0].message.content)
