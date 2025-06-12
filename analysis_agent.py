import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.exceptions import ModelHTTPError
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI
from openai.types import CreateEmbeddingResponse

load_dotenv()

llm_config = {"temperature": 0.02}
agent_retries = 2
agent_result_retries = 2

class AgentResponse(BaseModel):
    description: str
    code: str

class BossAgent:
    def __init__(self, llm_model=None):
        if llm_model is None:
            llm_model = self._get_llm()

        self.client = llm_model.client
        self.llm_model = llm_model

        self.boss_agent = Agent(
            model=llm_model,
            system_prompt=(
                "You are the Boss Agent. Your job is to analyze WiFi driver log files and detect connectivity issues related only to Intel WiFi component.\n"
                "Upon any textual log chunk, invoke the following tools by order:\n"
                "1) detect_user_flows: identify scenario/flow from log\n"
                "2) analyze_with_context: compare the flow indentified with known connectivity requirements\n"
                "Based on the log provided, the flows detected, and the context retrieved, decide if the end user experienced issues with connection to WiFi that might indicate "
                "a potential bug or wrong implementation in the Intel WiFi component. "
                "Try to filter out environment issues (like no connection due to no response from third party) or cases where we see multiple roaming but they "
                "might be just when someone walk through an office corridor.\n"
                "Always return your aswer as: 'Connectivity Issue Detected: Yes/No\nReason: <reasoning for your answer - include evidence from provided log>'."
            ),
            model_settings=llm_config,
            retries=agent_retries,
            result_retries=agent_result_retries
        )

        self.vector_client = self._init_vector_search()
        self.flow_detection_agent = self._init_flow_detection_agent()
        self.rag_agent = self._init_rag_agent()

        self._register_tools()

    def _get_llm(self):
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        return OpenAIModel("gpt-4o", provider=OpenAIProvider(openai_client=azure_client))

    def _init_vector_search(self):
        return SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT_200"),
            index_name="wifi-connectivity",
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY_200"))
        )

    def _init_flow_detection_agent(self):
        return Agent(
            model=self.llm_model,
            result_type=AgentResponse,
            system_prompt=(
                "You are an AI assistant tasked with analyzing log entries related to Wi-Fi connections to identify user flows.\n"
                "Format: [Timestamp] (Line Number) Event - SSID_MAC: Line Content\n"
                "Detect flows like: AP selection, Roaming, Connection Flow, Staying Connected."
            ),
            model_settings=llm_config,
            retries=agent_retries,
            result_retries=agent_result_retries
        )

    def _init_rag_agent(self):
        return Agent(
            model=self.llm_model,
            result_type=AgentResponse,
            system_prompt=(
                "You are a helpful assistant. Given a WiFi user flow and context from documentation, determine if there are issues in the Intel WiFi component. "
                "Try to filter out environment issues (like no connection due to no response from third party) or cases where we see multiple roaming but they "
                "might be just when someone walk through an office corridor. Be concise."
            ),
            model_settings=llm_config,
            retries=agent_retries,
            result_retries=agent_result_retries
        )

    async def _embed_text(self, text: str):
        response: CreateEmbeddingResponse = await self.client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def _register_tools(self):
        @self.boss_agent.tool
        async def detect_user_flows(ctx: RunContext, log_content: str) -> str:
            print(f"[DEBUG] detect_user_flows")
            response = await self.flow_detection_agent.run(log_content)
            return response

        @self.boss_agent.tool
        async def analyze_with_context(ctx: RunContext, user_flow: str) -> str:
            print(f"[DEBUG] analyze_with_context")

            embedding = await self._embed_text(user_flow)
            vector = VectorizedQuery(
                vector=embedding, 
                k_nearest_neighbors=8,
                fields="content_vector"
            )

            results = self.vector_client.search(
                search_text="",
                vector_queries=[vector],
                select=["content"]
            )
            content_items = []
            for doc in results:
                if "content" in doc:
                    content_items.append(doc["content"])
            context_chunks = "\n".join(content_items)

            prompt = f"Context:\n{context_chunks}\n\nUser Flow:\n{user_flow}\n\n"
            response = await self.rag_agent.run(prompt)
            return response

    async def run_with_retry_async(self, user_query: str, max_retries: int = 3, backoff: int = 2) -> dict:
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.boss_agent.run(user_query)
                return response.data
            except ModelHTTPError as e:
                if "429" in str(e):
                    if attempt == max_retries:
                        raise
                    sleep_secs = backoff ** attempt
                    print(f"Rate limit error. Retrying in {sleep_secs} seconds...")
                    await asyncio.sleep(sleep_secs)
                else:
                    raise
        return {"description": "", "result": "Failed after retries."}

    async def run_async(self, user_query: str) -> dict:
        response = await self.run_with_retry_async(user_query)
        return str(response.get("result", "")) if isinstance(response, dict) else {"result": str(response)}

    def run(self, user_query: str) -> dict:
        return asyncio.run(self.run_async(user_query))

async def main():
    boss_agent = BossAgent()
    res_path = os.path.join(os.path.dirname(__file__), '..', 'res')
    for root, dirs, files in os.walk(res_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                file_content = f.read()
                print(f"Processing file {file}.")
                response = await boss_agent.run_async(file_content)
                print(response['result'])


if __name__ == "__main__":
    asyncio.run(main())
