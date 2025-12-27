from openai import OpenAI
import uuid


client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Generate a unique ID
unique_id = str(uuid.uuid4())

response = client.chat.completions.create(  
  model="gpt-4o-mini", 
  messages=messages,
# Pass a user identification key
  user=unique_id
)

print(response.choices[0].message.content)
