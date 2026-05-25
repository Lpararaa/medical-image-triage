import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import datetime
import json

def json_log_callback(step_output):
    try:
        log_file = os.path.join(os.path.dirname(__file__), "..", "agent_logs.jsonl")
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action_output": str(step_output)
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception:
        pass

load_dotenv()

# 1. (Removed custom PneumoniaCNN class to align dictionary keys)

# 2. Define the Custom PyTorch Inference Tool
class PyTorchInferenceTool(BaseTool):
    name: str = "PyTorch Pneumonia Classifier"
    description: str = "Classify a chest X-ray image. Input should be the absolute or relative path to the image file (e.g., 'image.jpeg')."

    def _run(self, image_path: str) -> str:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            model = model.to(device)
            
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "pneumonia_resnet18.pth")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = Image.open(image_path).convert('RGB')
            img_t = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_t)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                prob_normal = probabilities[0].item()
                prob_pneumonia = probabilities[1].item()
            
            prediction = "PNEUMONIA" if prob_pneumonia > prob_normal else "NORMAL"
            confidence = prob_pneumonia if prediction == "PNEUMONIA" else prob_normal
            return f"Diagnosis: {prediction} (Confidence: {confidence*100:.2f}%)"
        except Exception as e:
            return f"Error classifying image: {str(e)}. Proceed by explicitly stating the image could not be analyzed due to technical issues."

# 3. Define the PubMed Tool Wrapper
class PubMedCustomTool(BaseTool):
    name: str = "PubMed Search Tool"
    description: str = "Search PubMed for medical research papers and clinical guidelines. Input should be a search query."
    
    def _run(self, query: str) -> str:
        try:
            pubmed = PubmedQueryRun()
            result = pubmed.run(query)
            return result
        except Exception as e:
            return f"Error executing PubMed search: {str(e)}. Please proceed with general medical knowledge if search is unavailable."

# 4. Define the Agents
def get_agents():
    # Initialize LLMs with LangChain fallback logic using available models
    primary_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)
    backup_llm_1 = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
    backup_llm_2 = ChatGoogleGenerativeAI(model="gemini-3.1-pro", temperature=0.1)
    
    # If the primary hits a rate limit, it automatically shifts to backup 1, then backup 2
    llm_with_fallback = primary_llm.with_fallbacks([backup_llm_1, backup_llm_2])

    diagnostician = Agent(
        role='Lead AI Diagnostician',
        goal='Analyze medical images using the PyTorch tool and provide accurate diagnoses.',
        backstory='You are a leading AI medical diagnostician. You use your specialized PyTorch vision model to interpret chest X-rays.',
        verbose=True,
        allow_delegation=False,
        tools=[PyTorchInferenceTool()],
        llm=llm_with_fallback,
        max_iter=3,
        max_retry_limit=2,
        step_callback=json_log_callback
    )

    researcher = Agent(
        role='Clinical Researcher',
        goal='Search for the most up-to-date medical guidelines and research based on the diagnosis.',
        backstory='You are a medical researcher who searches clinical databases for treatment guidelines and recommendations.',
        verbose=True,
        allow_delegation=False,
        tools=[PubMedCustomTool()],
        llm=llm_with_fallback,
        max_iter=3,
        max_retry_limit=2,
        step_callback=json_log_callback
    )

    orchestrator = Agent(
        role='Chief Medical Officer',
        goal='Synthesize diagnosis and research into a structured Medical Triage Report.',
        backstory='You are the Chief Medical Officer overseeing the triage process. You compile the work of your diagnostician and researcher into a final, professional report.',
        verbose=True,
        allow_delegation=True,
        llm=llm_with_fallback,
        max_iter=3,
        max_retry_limit=2,
        step_callback=json_log_callback
    )
    
    return diagnostician, researcher, orchestrator

# 5. Functions to be called by Streamlit
def generate_initial_report(image_path: str):
    diagnostician, researcher, orchestrator = get_agents()

    diagnosis_task = Task(
        description=f"Use your PyTorch tool to classify the chest X-ray located at: {image_path}. Return the exact diagnosis and confidence.",
        expected_output="A short sentence with the diagnosis and confidence score.",
        agent=diagnostician
    )

    research_task = Task(
        description="Take the diagnosis from the diagnostician. If the diagnosis is PNEUMONIA, search for standard medical guidelines and treatment protocols for Pneumonia. If NORMAL, search for standard discharge and wellness advice.",
        expected_output="A summary of clinical guidelines matching the diagnosis.",
        agent=researcher
    )

    current_date = datetime.date.today().strftime("%B %d, %Y")
    report_task = Task(
        description=f"Compile the preliminary diagnosis and the clinical guidelines into a structured 'Medical Triage Report'. The report MUST include a section for 'Physician Notes'. Make sure to use the current date in the report header: {current_date}.",
        expected_output="A markdown-formatted Medical Triage Report.",
        agent=orchestrator,
        human_input=False  # Disabled for Streamlit; handled via chat UI
    )

    crew = Crew(
        agents=[diagnostician, researcher, orchestrator],
        tasks=[diagnosis_task, research_task, report_task],
        process=Process.sequential,
        max_rpm=10
    )

    result = crew.kickoff()
    return result.raw

def revise_report(draft_report: str, feedback: str):
    _, _, orchestrator = get_agents()

    revision_task = Task(
        description=f"You are given a drafted Medical Triage Report and some feedback from the attending physician.\n\nDraft Report:\n{draft_report}\n\nPhysician Feedback: {feedback}\n\nRevise the report incorporating the feedback. Keep the professional formatting intact.",
        expected_output="The fully revised markdown-formatted Medical Triage Report.",
        agent=orchestrator
    )

    crew = Crew(
        agents=[orchestrator],
        tasks=[revision_task],
        process=Process.sequential,
        max_rpm=10
    )

    result = crew.kickoff()
    return result.raw
