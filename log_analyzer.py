from dotenv import load_dotenv
import os
import argparse
import tiktoken
import atexit
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables:
load_dotenv()

azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT_200')
azure_search_api_key = os.getenv('AZURE_SEARCH_API_KEY_200')
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_openai_deployment_name = os.getenv('AZURE_OPENAI_LLM_4')
azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')

# Initialize the AzureChatOpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=azure_openai_deployment_name,
    api_version=azure_openai_api_version,
    openai_api_key=azure_openai_api_key,
    temperature=0
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment="text-embedding-ada-002",
    openai_api_key=azure_openai_api_key,
    openai_api_version=azure_openai_api_version
)

# Initialize AzureSearch vector store
vector_store = AzureSearch(
    azure_search_endpoint=azure_search_endpoint,
    azure_search_key=azure_search_api_key,
    index_name="wifi-connectivity",
    embedding_function=embeddings.embed_query,
)

def analyze_chunk(log_content: str) -> str:
    docs = vector_store.similarity_search(log_content, k=8)
    context = "\n".join([doc.page_content for doc in docs])

    system_message = SystemMessage(content=(
        "You are an AI assistant tasked with analyzing log entries related to Wi-Fi connections in order to identify user connectivity issues. "
        "Each log entry follows this format:\n"
        "[Timestamp] (Line Number) Event - SSID_MAC: Line Content\n"
        "Use this format and based on the context provided determine if the user (human) experienced connection issues."
        "If connectivity issues found, return 'Connectivity issues found'."
        "If no connectivity issues found, return 'No connectivity issues found.' "
        "Return also what you understand from the log without reference to actual lines. "
    ))

    user_message = HumanMessage(content=(
        f"Context:\n{context}\n\nLog:\n{log_content}\n\n"
    ))

    response = llm([system_message, user_message])
    return response.content

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def cleanup():
    # Explicitly close the event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()

def main():
    total_tokens = 0

    res_path = os.path.join(os.path.dirname(__file__), '..', 'res')
    for root, dirs, files in os.walk(res_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                file_content = f.read()

                token_count = count_tokens(file_content)
                total_tokens += token_count
                print(f"Processing file {file} with {token_count} tokens.")
                response = analyze_chunk(file_content)
                print(response)

    print(f"Total tokens processed: {total_tokens}")

if __name__ == "__main__":
    atexit.register(cleanup)
    main()