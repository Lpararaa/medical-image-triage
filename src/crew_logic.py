import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from dotenv import load_dotenv
import datetime

load_dotenv()

# 1. Define the exact same PyTorch architecture as train.py
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.base_model = models.resnet18(weights=None)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# 2. Define the Custom PyTorch Inference Tool
class PyTorchInferenceTool(BaseTool):
    name: str = "PyTorch Pneumonia Classifier"
    description: str = "Classify a chest X-ray image. Input should be the absolute or relative path to the image file (e.g., 'image.jpeg')."

    def _run(self, image_path: str) -> str:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PneumoniaCNN().to(device)
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
                prob = output.item()
            
            prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"
            confidence = prob if prediction == "PNEUMONIA" else 1.0 - prob
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
    llm = LLM(model="gemini/gemini-3.1-flash-lite", temperature=0.1)

    diagnostician = Agent(
        role='Lead AI Diagnostician',
        goal='Analyze medical images using the PyTorch tool and provide accurate diagnoses.',
        backstory='You are a leading AI medical diagnostician. You use your specialized PyTorch vision model to interpret chest X-rays.',
        verbose=True,
        allow_delegation=False,
        tools=[PyTorchInferenceTool()],
        llm=llm,
        max_iter=3,
        max_retry_limit=2
    )

    researcher = Agent(
        role='Clinical Researcher',
        goal='Search for the most up-to-date medical guidelines and research based on the diagnosis.',
        backstory='You are a medical researcher who searches clinical databases for treatment guidelines and recommendations.',
        verbose=True,
        allow_delegation=False,
        tools=[PubMedCustomTool()],
        llm=llm,
        max_iter=3,
        max_retry_limit=2
    )

    orchestrator = Agent(
        role='Chief Medical Officer',
        goal='Synthesize diagnosis and research into a structured Medical Triage Report.',
        backstory='You are the Chief Medical Officer overseeing the triage process. You compile the work of your diagnostician and researcher into a final, professional report.',
        verbose=True,
        allow_delegation=True,
        llm=llm,
        max_iter=3,
        max_retry_limit=2
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
