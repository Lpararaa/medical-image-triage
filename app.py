import streamlit as st
import os
from src.crew_logic import generate_initial_report, revise_report
from markdown_pdf import MarkdownPdf, Section
import tempfile

st.set_page_config(page_title="Medical Image Triage System", page_icon="🩺", layout="wide")

# Custom CSS for aesthetic improvements
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00e676;
        font-family: 'Inter', sans-serif;
    }
    .stChatInput {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Medical Image Triage Agents")
st.markdown("Upload a Chest X-ray image to automatically trigger the Multi-Agent triage system (Diagnostician, Researcher, and Chief Medical Officer).")

# Initialize session state
if "report" not in st.session_state:
    st.session_state.report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "is_approved" not in st.session_state:
    st.session_state.is_approved = False

# Sidebar for Image Upload
with st.sidebar:
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Drag and drop an X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.image_path = temp_path
        st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
        
        if st.button("Run Agents 🚀"):
            st.session_state.report = None
            st.session_state.chat_history = []
            st.session_state.is_approved = False
            with st.spinner("Agents are analyzing the image and searching PubMed... (This may take ~1 minute)"):
                try:
                    report = generate_initial_report(st.session_state.image_path)
                    st.session_state.report = report
                    st.session_state.chat_history.append({"role": "assistant", "content": report})
                except Exception as e:
                    st.error(f"Error during agent execution: {str(e)}")

# Main content area for Chat / Report
if st.session_state.report:
    st.subheader("Final Medical Triage Report")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Approval and Download
    if st.session_state.is_approved:
        st.success("✅ Report Approved! The workflow is complete.")
        
        # Generate PDF
        try:
            import uuid
            pdf = MarkdownPdf(toc_level=0)
            pdf.add_section(Section(st.session_state.report, toc=False))
            
            tmp_pdf_name = os.path.join("temp_uploads", f"report_{uuid.uuid4().hex}.pdf")
            os.makedirs("temp_uploads", exist_ok=True)
            pdf.save(tmp_pdf_name)
                
            with open(tmp_pdf_name, "rb") as f:
                pdf_bytes = f.read()
                
            try:
                os.remove(tmp_pdf_name)
            except Exception:
                pass
                
            st.download_button(
                label="📄 Download Medical Report as PDF",
                data=pdf_bytes,
                file_name="Medical_Triage_Report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            
    # Feedback Chat Input
    elif feedback := st.chat_input("Provide feedback to revise the report, or type 'Looks good' to approve."):
        st.session_state.chat_history.append({"role": "user", "content": feedback})
        with st.chat_message("user"):
            st.markdown(feedback)
            
        # Check if the user is approving the report
        approval_keywords = ["looks good", "look good", "looks great", "approve", "approved", "ok", "yes", "perfect"]
        if any(keyword in feedback.lower() for keyword in approval_keywords) and len(feedback) < 30:
            st.session_state.is_approved = True
            st.rerun()
        else:
            # Run revision agent
            with st.chat_message("assistant"):
                with st.spinner("Chief Medical Officer is revising the report..."):
                    try:
                        revised_report = revise_report(st.session_state.report, feedback)
                        st.session_state.report = revised_report
                        st.session_state.chat_history.append({"role": "assistant", "content": revised_report})
                        st.markdown(revised_report)
                    except Exception as e:
                        st.error(f"Error during revision: {str(e)}")
else:
    st.info("Upload an image and click 'Run Agents' in the sidebar to begin.")
