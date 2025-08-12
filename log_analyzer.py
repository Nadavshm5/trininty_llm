from dotenv import load_dotenv
import os
import json
import tiktoken
import atexit
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
import time
import re
from openai import AzureOpenAI

# Load environment variables:
load_dotenv()

azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT_200')
azure_search_api_key = os.getenv('AZURE_SEARCH_API_KEY_200')
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_openai_deployment_name = os.getenv('AZURE_OPENAI_LLM_4')
azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')

CONTEXT_METHOD = "Assistant" # "RAG"

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

assistant_instructions = '''
You are a WiFi validation engineer, familiar with different WiFi connectivity flow.
the user will send you different user-flows (e.g. roaming, AP selection, etc.) in the prompt. 
your job is to describe the correct flow described by vectorstore context and general knowladge on latest 802.11 IEEE SPEC.
'''

client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_version=azure_openai_api_version,
    api_key=azure_openai_api_key,
)
assistant = client.beta.assistants.create(
    model=azure_openai_deployment_name,
    name="ConnectivityAssistant",
    instructions=assistant_instructions,
    tools=[{"type":"file_search"}],
    tool_resources={"file_search":{
        "vector_store_ids":["vs_XcYl6BTDUX3zfLmPFijTGfQU"]}},
    temperature=0.2,
    top_p=1
)

def run_assistant(prompt: str) -> str:
    # 1) start a thread + send user message
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=prompt
    )
    # 2) kick off a run (the assistant’s generation)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    # 3) poll until done
    while run.status in ("queued","in_progress","cancelling"):
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    if run.status=="completed":
        msgs = client.beta.threads.messages.list(thread_id=thread.id)
        # return the assistant’s reply
        return next((m.content for m in msgs if m.role=="assistant"), "")
    return ""

def get_context_per_event(log_content: str) -> str:
    all_events = set()
    for line in log_content.splitlines():
        match = re.search(r'\[.*?\]\s*\(.*?\).*?(\[.*)', line)
        if match:
            line = match.group(1)
        all_events.add(line)

    context = ""
    for event in all_events:
        if CONTEXT_METHOD == "RAG":
            docs = vector_store.similarity_search(event, k=8)
            context += "\n".join([doc.page_content for doc in docs])
        elif CONTEXT_METHOD == "Assistant":
            context = run_assistant(event)
        context += "\n"
    return context

def get_context_per_log(log_content: str) -> str:
    # clean_log_lines = []
    # for line in log_content.splitlines():
    #     match = re.search(r'\[.*?\]\s*\(.*?\).*?(\[.*)', line)
    #     if match:
    #         line = match.group(1)
    #     clean_log_lines.append(line)
    # clean_log = "\n".join(clean_log_lines)
    clean_log = log_content

    context = ""
    if CONTEXT_METHOD == "RAG":
        docs = vector_store.similarity_search(clean_log, k=8)
        context += "\n".join([doc.page_content for doc in docs])
    elif CONTEXT_METHOD == "Assistant":
        context = run_assistant(clean_log)
    return context

def analyze_chunk(log_content: str) -> str:
    context = get_context_per_log(log_content)
    print('-' * 80)
    print(f'RAG context:\n{context}')
    print('-' * 80)

    default_system_prompt = (
        "Objective: You are an intelligent agent designed to analyze WiFi driver logs to identify connectivity issues specifically related to the driver."
        "Each log entry follows this format:\n"
        "[Timestamp] (Line Number) Event - SSID_MAC: Line Content\n"
        "Your analysis should prioritize issues based on the following hierarchy:"
        "WRT Issue"
        "Limited Connectivity"
        "Borderline conditions - RF"
        "Assert"
        "PoorlyDisc"
        "Missing Debug Data"
        "Bad Peer Behavior"
        "Wrong Prediction"
        "Bug"
        "unknown scenario"
        "if couple of issues observed in the same log , determine the final classification according to priority."
        "Resources: You have access to a Retrieval-Augmented Generation (RAG) file containing comprehensive information on WiFi driver system requirements and specifications."
        "Use this resource to inform your analysis and ensure accuracy in identifying driver-related issues."
        "Instructions:"
        "Log Analysis:"
        "Review the WiFi driver logs provided."
        "Identify patterns, anomalies, or error codes that indicate potential connectivity issues."
        "Issue Classification:"
        "Determine if the connectivity issue is directly related to the WiFi driver."
        "Your classification options are: 'Borderline conditions - RF', 'Wrong Prediction', 'Environment', 'Driver Bug', 'Inapplicable', 'Missing Debug Data', 'Wrong Detection'."
        "Prioritization:"
        "1. WRT Issue: If scenarios such as [WRT2G] observed, classify as 'Inapplicable' and ignore other conditions."
        "2. Limited Connectivity: If limited connectivity is observed, classify as 'Environment'."
        "3. Borderline conditions - RF: RSSI values are inherently negative and should be treated as such. Ensure that the '-' sign is interpreted as a minus sign indicating a negative value. "
        "If the majority of 'BC 0' prints show RSSI values lower than -78 dB (e.g., -79, -80), classify as 'Borderline conditions - RF'."
        "Example - An average RSSI level of -65 is considered good, while an average RSSI level of -79, -80, -81, and so on is considered bad."
        "4. assert: when a fatal error or FW assert observed , classify as assert"
        "5. PoorlyDisc: If 'PoorlyDisc' value is '25' is seen more than once, classify as 'Environment' and mention the AP is probably not seen in scan. If 'PoorlyDisc' value is '100', it is normal."
        "6. Missing Debug Data: For issues like Auth Tx failure or assoc Tx failure AND when the average RSSI level are within acceptable levels , meaning average rssi above -78"
        "classify as 'Missing Debug Data' and recommend additional air sniffer, if the average RSSI is lower, prefer classification of 'Borderline conditions - RF' "
        "7. Bad Peer Behavior: Identify issues arising from other devices or network participants affecting connectivity."
        "8. Wrong Prediction: If no connectivity issue or problem is observed, classify as 'Wrong Prediction'."
        "9. Bug: Identify any driver-related bugs."
        "10. Unknown scenario : if non of the above seems to fit , please classify as unknown scenario"
        "Utilization of RAG File:"
        "Reference the RAG file to verify driver specifications and requirements."
        "Use the information to support your analysis and ensure that identified issues align with known driver limitations or requirements."
        "Reporting:"
        "Provide a clear and concise report of your findings."
        "Include a summary of identified driver-related issues and an explanation of issues attributed to external factors."
        "Continuous Improvement:"
        "Learn from each analysis to improve future assessments."
        "Adapt your approach based on feedback and new information."
        "Output Format:"
        "Use structured data formats (e.g., JSON, CSV) for easy integration with other systems."
        "Ensure clarity and precision in your language to facilitate understanding by technical teams."
        "Additional Considerations:"
        "Maintain confidentiality and security of log data."
        "Ensure compliance with relevant data protection regulations."
    )
    system_message = SystemMessage(content=default_system_prompt)

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
    # close vector store & asyncio loop
    try:
        vector_store.client.close()
    except:
        pass
    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()

def main():
    total_tokens = 0

    res_path = os.path.join(os.path.dirname(__file__), 'res')
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