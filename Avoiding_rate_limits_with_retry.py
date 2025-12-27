# Import the tenacity library
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import openAI

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Add the appropriate parameters to the decorator
# retry decorators with the parameters required to start retrying at an interval of 5 seconds, up to 40 seconds, and to stop after 4 attempts
@retry(wait=wait_random_exponential(min=5,max=40), stop=stop_after_attempt(4))
def get_response(model, message):
    response = client.chat.completions.create(
      model=model,
      messages=[message]
    )
    return response.choices[0].message.content
print(get_response("gpt-4o-mini", {"role": "user", "content": "List ten holiday destinations."}))
