import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Setup Logging (JSON format as required)
LOG_FILE = "logs/agent_actions.json"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_action(agent_name, action, result):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "result": str(result)
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# 2. Define the Custom PyTorch Tool
class PyTorchInferenceTool(BaseTool):
    name: str = "PyTorch Pneumonia Classifier"
    description: str = "Use this tool to classify a chest X-ray image. Provide the absolute or relative path to the image file as input."
    model_path: str = Field(default="models/pneumonia_resnet18.pth")
    
    def _run(self, image_path: str) -> str:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            class_names = ['NORMAL', 'PNEUMONIA']
            
            # Recreate model architecture
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            
            # Load weights
            model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
            model = model.to(device)
            model.eval()

            # Prepare image
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()

            result = f"Diagnosis: {class_names[preds[0]]} (Confidence: {confidence:.2%})"
            log_action("PyTorch Tool", f"Classified image: {image_path}", result)
            return result
        except Exception as e:
            error_msg = f"Error classifying image: {str(e)}"
            log_action("PyTorch Tool", f"Failed to classify image: {image_path}", error_msg)
            return error_msg

# 3. Define the PubMed Tool Wrapper
class PubMedCustomTool(BaseTool):
    name: str = "PubMed Search Tool"
    description: str = "Search PubMed for medical research papers and clinical guidelines. Input should be a search query."
    
    def _run(self, query: str) -> str:
        pubmed = PubmedQueryRun()
        result = pubmed.run(query)
        log_action("PubMed Tool", f"Searched for: {query}", result[:100] + "...")
        return result

# 4. Define the LLM backend
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY not found in environment. Please add it to your .env file.")

# Use CrewAI's native LLM wrapper (which uses LiteLLM under the hood)
llm = LLM(model="gemini/gemini-3.1-flash-lite", temperature=0.1)

def agent_step_callback(step_output):
    log_action("Agent", "Step executed", step_output)


# 4. Define Agents
diagnostician = Agent(
    role='Lead AI Diagnostician',
    goal='Accurately classify medical images using the PyTorch model',
    backstory='You are an expert radiologist AI. You use deep learning tools to analyze X-rays and provide preliminary diagnoses.',
    verbose=True,
    allow_delegation=False,
    tools=[PyTorchInferenceTool()],
    llm=llm,
    step_callback=agent_step_callback
)

researcher = Agent(
    role='Clinical Researcher',
    goal='Find the latest clinical guidelines and next steps based on a diagnosis',
    backstory='You are a medical researcher who searches clinical databases for treatment guidelines and recommendations.',
    verbose=True,
    allow_delegation=False,
    tools=[PubMedCustomTool()],  # Wrapped PubMed Tool!
    llm=llm,
    step_callback=agent_step_callback
)

orchestrator = Agent(
    role='Chief Medical Officer (Orchestrator)',
    goal='Compile the diagnosis and research into a final structured medical report',
    backstory='You are the head of the medical department. You take inputs from your specialists and write comprehensive, easy-to-read reports for patients and human doctors.',
    verbose=True,
    allow_delegation=True,
    llm=llm,
    step_callback=agent_step_callback
)

# 5. Define Tasks and Workflow
def run_triage(image_path):
    print(f"\n--- Starting Triage Workflow for: {image_path} ---")
    
    diagnosis_task = Task(
        description=f'Use your PyTorch tool to classify the chest X-ray located at: {image_path}. Return the exact diagnosis and confidence.',
        expected_output='A preliminary diagnosis with confidence score.',
        agent=diagnostician
    )

    research_task = Task(
        description='Take the diagnosis from the diagnostician. If the diagnosis is PNEUMONIA, search for standard medical guidelines and treatment protocols for Pneumonia. If NORMAL, search for standard discharge and wellness advice.',
        expected_output='A summary of clinical guidelines for the specific condition.',
        agent=researcher,
        context=[diagnosis_task]
    )

    report_task = Task(
        description='Compile the preliminary diagnosis and the clinical guidelines into a structured "Medical Triage Report". The report MUST include a section for "Physician Notes" and ask the user to approve.',
        expected_output='A highly professional, formatted markdown medical report.',
        agent=orchestrator,
        context=[diagnosis_task, research_task],
        human_input=True  # Human-in-the-loop checkpoint!
    )

    crew = Crew(
        agents=[diagnostician, researcher, orchestrator],
        tasks=[diagnosis_task, research_task, report_task],
        process=Process.sequential,
        max_rpm=10
    )

    result = crew.kickoff()
    return result

if __name__ == "__main__":
    # Test on a specific image from our dataset
    test_image = "chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg" 
    
    if not os.path.exists(test_image):
        print(f"Test image not found at {test_image}. Using a fallback image if available.")
        test_image = "chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
        
    final_report = run_triage(test_image)
    
    print("\n\n" + "="*50)
    print("FINAL APPROVED MEDICAL REPORT")
    print("="*50)
    print(final_report)
