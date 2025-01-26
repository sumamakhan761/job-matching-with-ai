from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
import uuid
from langchain_groq import ChatGroq
import streamlit as st
import chromadb
from bs4 import BeautifulSoup
import PyPDF2
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ"),
    model_name="llama-3.3-70b-versatile"
)

def preprocess_job_posting(url, portfolio_file):
    try:
        # Check if the uploaded file is a CSV or PDF
        if portfolio_file.name.endswith('.csv'):
            df = pd.read_csv(portfolio_file)
        elif portfolio_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(portfolio_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

            # Convert PDF text into a DataFrame assuming one skill per line
            data = [line.strip() for line in pdf_text.split("\n") if line.strip()]
            df = pd.DataFrame(data, columns=['Technology'])
        else:
            return {"error": "Unsupported file format. Please upload a CSV or PDF file."}

        # Scrape job posting
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content

        # Extract job details using the LLM
        prompt_extract = PromptTemplate.from_template("""
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the
            following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
        """)
        chain_extract = prompt_extract | llm
        res_1 = chain_extract.invoke(input={'page_data': page_data})

        # Parse the JSON response
        json_parser = JsonOutputParser()
        json_res = json_parser.parse(res_1.content)

        # Prepare ChromaDB collection
        client = chromadb.PersistentClient('vectorstore')
        collections = client.get_or_create_collection(name="technology_table")
        if not collections.count():
            for _, row in df.iterrows():
                collections.add(documents=row['Technology'], ids=[str(uuid.uuid4())])

        # Match skills and generate interview questions
        job = json_res.get('skills', []) if type(json_res) == dict else json_res[0].get('skills', [])

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
                - `skills_match`: A dictionary where each key is a skill, and the value is the matching percentage.
                - `interview_questions`: A list of tailored questions related to the job description.

            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
        """)
        chain_skills_and_question = prompt_skills_and_question | llm
        res2 = chain_skills_and_question.invoke({"job_description": str(job)})

        # Parse the final result
        final_result = json_parser.parse(res2.content)
        return final_result

    except Exception as e:
        return {"error": str(e)}

# Streamlit App
st.title("Job Scraping & Analyzer with Interview Preparation Questions")

# Input fields
url = st.text_input("Website URL", placeholder="Enter the URL of the job posting")
portfolio_file = st.file_uploader("Upload Portfolio File (CSV or PDF)", type=["csv", "pdf"])

if st.button("Analyze Job Posting"):
    if url and portfolio_file:
        # Process the input
        result = preprocess_job_posting(url, portfolio_file)

        # Display the result
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.subheader("Analysis Result")

            # Display skills match in a separate container
            st.subheader("Skills Match")
            skills_match = result.get('skills_match', {})
            container = st.container(border=True)
            for skill, match in skills_match.items():
                container.write(f"{skill}: {match}% match")

            # Display interview questions in another container
            st.subheader("Interview Questions")
            interview_questions = result.get('interview_questions', [])
            container = st.container(border=True)
            for question in interview_questions:
                container.write(f"- {question}")
    else:
        st.warning("Please provide both the URL and upload a valid file (CSV or PDF).")