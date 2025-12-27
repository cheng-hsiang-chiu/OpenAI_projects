from openai import OpenAI
import json

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Define the function to pass to tools
function_definition = [{"type": "function",
                        "function" : {"name": "get_airport_info",
                                      "description": "Convert the user request into airport codes",
                                      "parameters": {"type": "object", 
                                                     "properties": {"airport_code": {"type":"string",      
                                                                                     "description":"code"
                                                                                    }
                                                                   }
                                                    }, 
                                      "result":{"type":"string"} 
                                     }
                       }
                      ]


# Call the Chat Completions endpoint 
def get_response(function_definition):
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"system",
               "content":"You are an AI assistant, a specialist in aviation. You should be aware that it is the aviation space and that you need to extract the corresponding airport code based on the user input."},
              {"role": "user", 
               "content": "I'm planning to land a plane in JFK airport in New York and would like to have the corresponding information."}
             ],
    tools=function_definition
  )
  return response


response = get_response(function_definition)

if response.choices[0].finish_reason=='tool_calls':
  function_call = response.choices[0].message.tool_calls[0].function
  # Check function name
  if function_call.name == "get_airport_info":
    # Extract airport code
    code = json.loads(function_call.arguments)["airport code"]
    airport_info = get_airport_info(code)
    print(airport_info)
  else:
    print("Apologies, I couldn't find any airport.")
else: 
  print("I am sorry, but I could not understand your request.")
