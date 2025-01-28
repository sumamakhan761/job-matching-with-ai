# Import necessary libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
import uuid
from langchain_groq import ChatGroq
import gradio as gr  # Changed from Streamlit to Gradio
from langchain_chroma import Chroma
import chromadb
import PyPDF2
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the language model
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ"),
    model_name="llama-3.3-70b-versatile"
)

def preprocess_job_posting(url, portfolio_file):
    try:
        if not portfolio_file:
            return {"error": "No portfolio file uploaded."}
        
        if portfolio_file.name.endswith('.csv'):
            df = pd.read_csv(portfolio_file)
        elif portfolio_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(portfolio_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text

            data = [line.strip() for line in pdf_text.split("\n") if line.strip()]
            df = pd.DataFrame(data, columns=['Technology'])
        else:
            return {"error": "Unsupported file format. Please upload a CSV or PDF file."}

        # Load and scrape the job postings from the provided URL
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content

        # Prompt to extract job postings in JSON format
        prompt_extract = PromptTemplate.from_template("""
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the
            following keys: role, experience, skills and description.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
        """)
        chain_extract = prompt_extract | llm
        res_1 = chain_extract.invoke(input={'page_data': page_data})

        # Parse the JSON response
        json_parser = JsonOutputParser()
        json_res = json_parser.parse(res_1.content)

        # Initialize Chroma vector store
        client = chromadb.PersistentClient('vectorstore')
        collections = client.get_or_create_collection(name="technology_table")
        if not collections.count():
            for _, row in df.iterrows():
                collections.add(documents=row['Technology'], ids=[str(uuid.uuid4())])

        # Extract skills from the job postings
        job = json_res.get('skills', []) if isinstance(json_res, dict) else (json_res[0].get('skills', []) if json_res else [])

        # Prompt to analyze skills and generate interview questions
        prompt_skills_and_question = PromptTemplate.from_template("""
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mishu Dhar Chando, the CEO of Knowledge Doctor, a YouTube channel specializing in educating individuals on machine learning, deep learning, and natural language processing.
            Your expertise lies in bridging the gap between theoretical knowledge and practical applications through engaging content and innovative problem-solving techniques.
            Your job is to:
            1. Analyze the given job description to identify the required technical skills and match them with the provided skill set to calculate a percentage match.
            2. Generate a list of relevant interview questions based on the job description.
            3. Return the information in JSON format with the following keys:
                - skills_match: A dictionary where each key is a skill, and the value is the matching percentage.
                - interview_questions: A list of tailored questions related to the job description.

            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
        """)
        chain_skills_and_question = prompt_skills_and_question | llm
        res2 = chain_skills_and_question.invoke({"job_description": str(job)})

        # Parse the final JSON response
        final_result = json_parser.parse(res2.content)
        return final_result

    except Exception as e:
        return {"error": str(e)}

# Define the Gradio interface
def analyze_job_posting(url, portfolio_file):
    result = preprocess_job_posting(url, portfolio_file)
    
    if "error" in result:
        error_message = f"Error: {result['error']}"
        return error_message, "", ""
    else:
        # Prepare Skills Match output
        skills_match = result.get('skills_match', {})
        if isinstance(skills_match, dict):
            skills_output = "\n".join([f"{skill}: {match}% match" for skill, match in skills_match.items()])
        else:
            skills_output = "No skills match data available."

        # Prepare Interview Questions output
        interview_questions = result.get('interview_questions', [])
        if isinstance(interview_questions, list):
            questions_output = "\n".join([f"- {question}" for question in interview_questions])
        else:
            questions_output = "No interview questions available."

        return "", skills_output, questions_output

# Create Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 align='center'>Job Scraping & Analyzer with Interview Preparation Questions</h1>")

    with gr.Row():
        url_input = gr.Textbox(
            label="Website URL",
            placeholder="Enter the URL of the job posting",
            lines=1
        )

    with gr.Row():
        file_input = gr.File(
            label="Upload Portfolio File (CSV or PDF)",
            file_types=['.csv', '.pdf']
        )

    analyze_button = gr.Button("Analyze Job Posting")

    with gr.Accordion("Analysis Result", open=False):
        with gr.Column():
            gr.Markdown("### Skills Match")
            skills_output = gr.Textbox(
                label="Skills Match",
                lines=10,
                interactive=False
            )
            
            gr.Markdown("### Interview Questions")
            questions_output = gr.Textbox(
                label="Interview Questions",
                lines=10,
                interactive=False
            )

    with gr.Row():
        error_output = gr.Markdown("", visible=False)

    # Define the button click event
    analyze_button.click(
        fn=analyze_job_posting,
        inputs=[url_input, file_input],
        outputs=[error_output, skills_output, questions_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()