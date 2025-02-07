
# Using the retriaval qa prompt template from the langchain library - https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/retrieval_qa/prompt.py
def create_prompt(query, relevant_docs):
    relevant_text = ''
    relevant_docs = relevant_docs['documents'][0]
    for docs in relevant_docs:
        relevant_text += ("\n" + str(docs))

    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question. 
If you find the answer in the context, start with "Based on the provided documents, ". 
If you cannot find the exact answer, start with "I apologize, but based on the provided documents, ".

Context:
{relevant_text}

Question: {query}
Assistant:"""
    
    return prompt