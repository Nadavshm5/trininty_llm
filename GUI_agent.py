import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import asyncio
import json
import pandas as pd
from datetime import datetime
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
    def __init__(self, llm_model=None, system_prompt=None):
        if llm_model is None:
            llm_model = self._get_llm()

        self.client = llm_model.client
        self.llm_model = llm_model

        if system_prompt is None:
            system_prompt = ()

        self.boss_agent = Agent(
            model=llm_model,
            system_prompt=system_prompt,
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

    async def run_with_retry_async(self, user_query: str, max_retries: int = 3, backoff: int = 2) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.boss_agent.run(user_query)
                return response.output
            except ModelHTTPError as e:
                if e.status_code == 429:
                    retry_after = e.body.get('retry_after')
                    if retry_after:
                        sleep_secs = int(retry_after)
                    else:
                        sleep_secs = backoff ** attempt
                    print(f"Rate limit error. Retrying in {sleep_secs} seconds...")
                    await asyncio.sleep(sleep_secs)
                else:
                    raise
        return "Failed after retries."

    async def run_async(self, user_query: str) -> dict:
        response_text = await self.run_with_retry_async(user_query)

        # Debug: Print the response text to verify its format
        print("Response Text:", response_text)

        # Remove markdown code block indicators if present
        if response_text.startswith("```json"):
            response_text = response_text.strip("```json").strip("```").strip()

        # Parse the JSON response
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
            response_data = {}

        connectivity_issue = response_data.get("Connectivity Issue Detected","")
        reason = response_data.get("Reason", "")
        classification = response_data.get("Classification", "")
        Average_RSSI = response_data.get("Average RSSI", "")

        # Debug: Print parsed values to verify correctness
        #print("Parsed Connectivity Issue:", connectivity_issue)
        #print("Parsed Reason:", reason)
        #print("Parsed Classification:", classification)

        return {
            "connectivity_issue": connectivity_issue,
            "reason": reason,
            "classification": classification,
            "Average RSSI": Average_RSSI
        }

    def run(self, user_query: str) -> dict:
        return asyncio.run(self.run_async(user_query))


def parse_log_file(log_file_path, start_line, end_line):
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        config_dict = json.load(config_file)
        patterns_dict = config_dict['patterns']

    patterns = {event: re.compile(pattern) for event, pattern in patterns_dict.items()}
    ssid_mac_pattern = re.compile(r'(\w+) (\w{2}:\w{2}:\w{2}:\w{2}:\w{2}:\w{2})')

    log_data = []
    last_ssid_mac = None

    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if start_line <= line_number <= end_line:
                ssid_mac_match = ssid_mac_pattern.search(line)
                if ssid_mac_match:
                    last_ssid_mac = ssid_mac_match.group(0)

                for event, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        timestamp_str = match.group(1)
                        timestamp = datetime.strptime(timestamp_str, '%m/%d/%Y-%H:%M:%S.%f')
                        log_data.append((timestamp, event, last_ssid_mac, line_number, line.strip()))

    df = pd.DataFrame(log_data, columns=['Timestamp', 'Event', 'SSID_MAC', 'LineNumber', 'LineContent'])
    return df


def generate_text_representation(df):
    text_representation = []
    seen_timestamps = set()

    for index, row in df.iterrows():
        timestamp = row['Timestamp']
        event = row['Event']

        # Check if the event is "AP selection" and if the timestamp has been seen
        if event == "AP selection":
            if timestamp in seen_timestamps:
                continue  # Skip if this timestamp has already been processed for "AP selection"
            seen_timestamps.add(timestamp)

        # Add the line to the text representation
        text_representation.append(
            f"[{row['LineContent']}"
       )

    return "\n".join(text_representation)

# The rest of your code remains unchanged

async def process_log(log_file_path, start_line, end_line, boss_agent):
    df = parse_log_file(log_file_path, start_line, end_line)
    text_representation = generate_text_representation(df)

    # Save to a file
    res_dir = os.path.join(os.path.dirname(__file__), 'res')
    os.makedirs(res_dir, exist_ok=True)
    txt_path = os.path.join(res_dir, 'log_text_representation.txt')
    with open(txt_path, "w") as text_file:
        text_file.write(text_representation)

    # Analyze the text representation using the provided BossAgent instance
    response = await boss_agent.run_async(text_representation)
    return response, txt_path

# The rest of your code remains unchanged


def run_gui():
    def on_submit():
        log_file_path = log_path_entry.get().strip()
        start_line = int(start_line_entry.get())
        end_line = int(end_line_entry.get())
        jira_ticket = jira_ticket_entry.get().strip()

        if not os.path.isfile(log_file_path):
            messagebox.showerror("Error", "Log file path is invalid.")
            return

        updated_system_prompt = system_prompt_text.get("1.0", tk.END).strip()
        final_system_prompt = updated_system_prompt + "\n" + uneditable_prompt

        # Ensure the final system prompt includes the uneditable_prompt
        print("Final System Prompt:", final_system_prompt)  # Debug: Print the final system prompt

        boss_agent = BossAgent(llm_model=None, system_prompt=final_system_prompt)
        asyncio.run(async_process(log_file_path, start_line, end_line, boss_agent, jira_ticket, final_system_prompt))

    # Define the uneditable prompt
    uneditable_prompt = (
        "Always return your answer in JSON format with the following fields:\n"
        "{\n"
        "  \"Connectivity Issue Detected\": \"Yes/No\",\n"
        "  \"Reason\": \"<reasoning for your answer - include evidence from provided log>\",\n"
        "  \"Classification\": \"<3rd party / environment / boundary RF conditions / Bug / inapplicable / missing debug data / wrong detection>\"\n"
        "  \"Average RSSI\": Average_RSSI"
        "}"
    )

    async def async_process(log_file_path, start_line, end_line, boss_agent, jira_ticket, final_system_prompt):
        response, txt_path = await process_log(log_file_path, start_line, end_line, boss_agent)

        connectivity_issue = response.get('connectivity_issue', '')
        reason = response.get('reason', '')
        classification = response.get('classification', '')
        Average_RSSI = response.get('Average RSSI','')


        connectivity_issue_text.delete("1.0", tk.END)
        connectivity_issue_text.insert(tk.END, connectivity_issue)
        reason_text.delete("1.0", tk.END)
        reason_text.insert(tk.END, reason)
        classification_text.delete("1.0", tk.END)
        classification_text.insert(tk.END, classification)
        Average_RSSI_text.delete("1.0", tk.END)
        Average_RSSI_text.insert(tk.END, Average_RSSI)
        link_label.config(text=f"Log Text Representation: {txt_path}")

        append_to_csv(jira_ticket, log_file_path, start_line, end_line, connectivity_issue, reason, classification,
                      final_system_prompt)

    def append_to_csv(jira_ticket, log_path, start_line, end_line, connectivity_issue, reason, classification,
                      prompt_used):
        import csv
        csv_file_path = os.path.join(os.path.dirname(__file__), 'results.csv')
        fieldnames = ['Jira Ticket', 'Log Path', 'Start Line', 'End Line', 'Connectivity Issue Detected', 'Reason',
                      'Classification', 'Prompt Used']

        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'Jira Ticket': jira_ticket,
                'Log Path': log_path,
                'Start Line': start_line,
                'End Line': end_line,
                'Connectivity Issue Detected': connectivity_issue,
                'Reason': reason,
                'Classification': classification,
                'Prompt Used': prompt_used
            })

    def load_csv():
        csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not csv_file_path:
            return

        df = pd.read_csv(csv_file_path)
        results = []

        for index, row in df.iterrows():
            jira_ticket_entry.delete(0, tk.END)
            jira_ticket_entry.insert(tk.END, row['Issue key'])

            start_line_entry.delete(0, tk.END)
            start_line_entry.insert(tk.END, row['start line'])

            end_line_entry.delete(0, tk.END)
            end_line_entry.insert(tk.END, row['end line'])

            log_path_entry.delete(0, tk.END)
            log_path_entry.insert(tk.END, row['log path'])

            updated_system_prompt = system_prompt_text.get("1.0", tk.END).strip()
            final_system_prompt = updated_system_prompt + "\n" + uneditable_prompt

            boss_agent = BossAgent(llm_model=None, system_prompt=final_system_prompt)
            response = asyncio.run(process_log(row['log path'], row['start line'], row['end line'], boss_agent))

            connectivity_issue = response[0].get('connectivity_issue', '')
            reason = response[0].get('reason', '')
            classification = response[0].get('classification', '')
            Average_rssi = response[0].get('Average_RSSI', '')

            results.append({
                'Issue key': row['Issue key'],
                'log path': row['log path'],
                'start line': row['start line'],
                'end line': row['end line'],
                'connectivity issue detected': connectivity_issue,
                'reason': reason,
                'classification': classification
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv('processed_results.csv', index=False)

    root = tk.Tk()
    root.title("Log Analyzer")

    root.columnconfigure(1, weight=1)
    root.rowconfigure(7, weight=1)
    root.rowconfigure(8, weight=1)

    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)

    size_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Window Size", menu=size_menu)
    size_menu.add_command(label="800x600", command=lambda: set_window_size(800, 600))
    size_menu.add_command(label="1024x768", command=lambda: set_window_size(1024, 768))
    size_menu.add_command(label="1280x1024", command=lambda: set_window_size(1280, 1024))

    tk.Label(root, text="Log File Path:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    log_path_entry = tk.Entry(root, width=50)
    log_path_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

    tk.Label(root, text="Start Line:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    start_line_entry = tk.Entry(root, width=20)
    start_line_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

    tk.Label(root, text="End Line:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
    end_line_entry = tk.Entry(root, width=20)
    end_line_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

    tk.Label(root, text="Jira Ticket (optional):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
    jira_ticket_entry = tk.Entry(root, width=50)
    jira_ticket_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

    tk.Label(root, text="System Prompt:").grid(row=4, column=0, padx=10, pady=5, sticky="nw")
    system_prompt_text = tk.Text(root, height=10, width=100)
    system_prompt_text.grid(row=4, column=1, padx=10, pady=5, sticky="nsew")
    default_system_prompt = (
                "Objective: You are an intelligent agent designed to analyze WiFi driver logs to identify connectivity issues specifically related to the driver."
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
    system_prompt_text.insert(tk.END, default_system_prompt)


    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.grid(row=5, column=0, columnspan=2, pady=10)

    load_csv_button = tk.Button(root, text="Load CSV", command=load_csv)
    load_csv_button.grid(row=6, column=0, columnspan=2, pady=10)

    link_label = tk.Label(root, text="", fg="blue", cursor="hand2")
    link_label.grid(row=7, column=0, columnspan=2, pady=5)

    tk.Label(root, text="Connectivity Issue Detected:").grid(row=8, column=0, padx=10, pady=5, sticky="nw")
    connectivity_issue_text = tk.Text(root, height=5, width=100)
    connectivity_issue_text.grid(row=8, column=1, padx=10, pady=5, sticky="nsew")

    tk.Label(root, text="Reason:").grid(row=9, column=0, padx=10, pady=5, sticky="nw")
    reason_text = tk.Text(root, height=10, width=100)
    reason_text.grid(row=9, column=1, padx=10, pady=5, sticky="nsew")

    tk.Label(root, text="Classification:").grid(row=10, column=0, padx=10, pady=5, sticky="nw")
    classification_text = tk.Text(root, height=5, width=100)
    classification_text.grid(row=10, column=1, padx=10, pady=5, sticky="nsew")

    tk.Label(root, text="Average RSSI:").grid(row=11, column=0, padx=10, pady=5, sticky="nw")
    Average_RSSI_text = tk.Text(root, height=5, width=100)
    Average_RSSI_text.grid(row=11, column=1, padx=10, pady=5, sticky="nsew")


    root.mainloop()


if __name__ == "__main__":
    run_gui()