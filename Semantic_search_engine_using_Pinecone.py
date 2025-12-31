# Using Pinecone and OpenAI to implement a semantic search engine


from openai import OpenAI
from Pinecone import Pinecone, ServerlessSpec
import panda as pd
import numpy as np
from uuid import uuid4

client = OpenAI(api_key="OpenAI api key")
pc = Pinecone(api_key="Pinecone api key")



# Create Pinecone's Serverless index
pc.create_index(
  name="semantic-search",
  dimension=1536,   # dimension of OpenAI's embeddings
  spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Connect to the index
index = pc.Index("semantic-search")


# squad_dataset.csv has the following content
# | id | text                                              | title             | 
# | 1  | Architecturally, the school has a Catholic cha... | University of ... |
# | 2  | The College of Engineering was established in.... | University of ... |
# | 3  | Following the disbandment of Destiny's Child in.. | Beyonce |
# | 4  | Architecturally, the school has a Catholic cha... | University of ... |


df = pd.read_csv("squad_dataset.csv")

batch_limit = 100

for batch in np.array_split(df, len(df)/batch_limit):
  metadatas = [{"text_id": row["id"], "text": row["text"], "title": row["title"]} for _, row in batch.iterrows()]
  texts = batch["text"].tolist()

  # Create a unique id for each text
  ids = [str(uuid4()) for _ in range(len(texts))]

  # Using OpenAI to create the embeddings
  response = client.embeddings.create(input = texts, model = "text-embedding-3-small")
  embeds = [np.array(x.embedding) for x in response.data]

  index.upsert(vectors=zip(ids, embeds, metadatas), namespace="squad_dataset")

print(index.describe_index_stats())


# Querying with Pinecone
query = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
query_response = client.embeddings.create(input = query, model = "text-embeddings-3-small")
query_emb = query_response.data[0].embedding

retrieved_docs = index.query(
  vector = query_emb,
  top_k = 3,
  namesapce = "squad_dataset",
  include_medadata = True
)

for result in retrieved_docs["matches"]:
  print(f"{round(result["score"], 2)}: {result["metadata"]["text"]}")
