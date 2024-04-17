import os 
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import textract

import keys
os.environ["OPENAI_API_KEY"] = keys.openaiapi_key()

def pdf_to_text(path_to_pdf):
    loader = PyPDFLoader(path_to_pdf)
    pages = loader.load_and_split()
    print(pages[0])
    
    doc = textract.process(path_to_pdf)
    with open('temp.txt', 'w') as f:
        f.write(doc.decode('utf-8'))
    
    with open('temp.txt', 'r') as f:
        text = f.read()
        
    return text


def text_to_chunks(text):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    def count_tokens(text:str) -> int:
        return len(tokenizer.encode(text))
    
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512, 
            chunk_overlap = 24, 
            length_function = count_tokens,
    )
    
    chunks = text_splitter.create_documents([text])
    
    return chunks


def vectorstore_output(chunks):
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings)

    return db


def get_openAI_response(db_output, query,chat_history):
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        
    docs = db_output.similarity_search(query)
    print(docs)
    chain.run(input_documents=docs, question=query)
    
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature = 0.1), db_output.as_retriever())
    
    result = qa({"question":query, "chat_history":chat_history})
    chat_history.append((query, result['answer']))

    return result['answer']


def main():
    
    history = []
    pdf_name = input("Enter PDF name: ")
    pdf_text = pdf_to_text(pdf_name)
    pdf_chunks = text_to_chunks(pdf_text)
    vector_db_output = vectorstore_output(pdf_chunks)
    query = input("Enter query: ")
    response = get_openAI_response(vector_db_output, query, history)
    print(response)

    
if __name__ == "__main__":
    main()
