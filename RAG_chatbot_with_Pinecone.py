# Using Pinecone and OpenAI to implement a RAG chatbot


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

youtube_df = pd.read_csv("youtube_rag_data.csv")

# youtube_rag_data.csv has the following content
# id  | blob | channel_id | end | published | start | text | title | url | 
# int | dict | str        | int | datatime  | int   | str  | str   | str | 



batch_limit = 100

for batch in np.array_split(youtube_df, len(youtube_df)/batch_limit):
  metadatas = [{"text_id": row["id"], "text": row["text"], "title": row["title"], "url": row["url"], "published": row[["published"]} for _, row in batch.iterrows()]
  texts = batch["text"].tolist()

  # Create a unique id for each text
  ids = [str(uuid4()) for _ in range(len(texts))]

  # Using OpenAI to create the embeddings
  response = client.embeddings.create(input = texts, model = "text-embedding-3-small")
  embeds = [np.array(x.embedding) for x in response.data]

  index.upsert(vectors=zip(ids, embeds, metadatas), namespace="youtube_rag_dataset")

print(index.describe_index_stats())

# Retrieval function
def retrieve(query, top_k, namespace, emb_model):
  query_response = client.embeddings.create(input = query, model = emb_model)
  query_emb = query_response.data[0].embedding
  retrieved_docs = 
  []
  sources = []
  
  docs = index.query(
    vector = query_emb,
    top_k = top_k,
    namespace = namespace,
    include_medadata = True
  )
  
  for doc in docs["matches"]:
    retrieved_docs.append(doc["metadata"]["text"])
    sources.append((doc["metadata"]["title"], doc["metadata"]["url"]))
  return retrieved_docs, sources

query = "How to build next-level Q&A with OpenAI"
documents, sources = retrieve(query, top_k=3, namespace="youtube_rag_dataset", emb_model="text-embedding-3-small")

# Prompt with context builder function
def prompt_with_context_builder(query, docs):
  delim = '\n\n---\n\n'
  prompt_start = 'Answer the question based on the context below. \n\nContext:\n'
  prompt_end = f'\n\nQuestion: {query}\n Answer:'
  prompt = prompt_start + delim.join(docs) + prompt_end
  return prompt
query = "How to build next-level Q&A with OpenAI"
prompt_with_context = prompt_with_context_builder(query, documents)


# Question-answering function
def question_answering(prompt, sources, chat_model):
  sys_prompt = "You are a helpful assistant that always answers questions."
  res = client.chat.completions.create(
    model=chat_model,
    message=[{"role": "system", "content": sys_prompt}, {"role":"user", "content":prompt}],
    temperature=0
  )
  answer = res.choices[0].message.content.strip()
  answer += "\n\nSources:"
  for source in sources:
    answer += "\n" + source[0] + ": " + source[1]
  return answer


query = "How to build next-level Q&A with OpenAI"
answer = question_answering(prompt_with_context, sources, chat_model="gpt-4o-mini")
