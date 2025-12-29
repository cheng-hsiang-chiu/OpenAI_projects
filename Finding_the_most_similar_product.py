from scipy.spatial import distance
from openai import OpenAI
import numpy as np


# product has the following format
#products = [
#    {
#        "title": "Smartphone X1",
#        "short_description": "The latest flagship smartphone with AI-powered features and 5G connectivity.",
#        "price": 799.99,
#        "category": "Electronics",
#        "features": [
#            "6.5-inch AMOLED display",
#            "Quad-camera system with 48MP main sensor",
#            "Face recognition and fingerprint sensor",
#            "Fast wireless charging"
#        ]
#    },
#...
#]




# Define a create_embeddings function
def create_embeddings(texts):
  response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
  )
  response_dict = response.model_dump()
  
  return [data['embedding'] for data in response_dict['data']]

# Embed short_description and print
print(create_embeddings(short_description)[0])

# Embed list_of_descriptions and print
print(create_embeddings(list_of_descriptions)[0])



# Embed the search text
search_text = "soap"
search_embedding = create_embeddings(search_text)[0]

distances = []
for product in products:
  # Compute the cosine distance for each product description
  dist = distance.cosine(search_embedding, product['embedding'])
  distances.append(dist)

# Find and print the most similar product short_description    
min_dist_ind = np.argmin(distances)
print(products[min_dist_ind]['short_description'])
