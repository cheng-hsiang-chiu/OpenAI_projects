from openai import OpenAI

# Create an OpenAI client
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Create a request to obtain embeddings
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="what is the embedding?"
)

# Convert the response into a dictionary
response_dict = response.model_dump()

print(response_dict)
