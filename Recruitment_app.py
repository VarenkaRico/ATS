# streamlit_recruitment_app.py
import streamlit as st
import boto3
import tempfile
import os
from botocore.exceptions import NoCredentialsError

import uuid

from dotenv import load_dotenv

from utils.job_description import generate_job_description, split_job_description_sections, get_section_embeddings_dict, save_job_description_to_json, validate_job_description
from utils.general_funcs import upload_to_s3, detect_language, generate_clean_html, generate_pdf_from_html
from utils.resume import extract_resume_sections, embed_resume_sections, format_section_for_display, save_resume_to_json,parse_resume_to_text
from utils.matching import load_job_descriptions_list, get_job_chunks, get_matching_resumes, rank_resumes_for_job, compare_job_resume_embeddings,plot_job_based_radar_multi
from utils.testing import generate_test_from_description
from utils.logger import log_event

load_dotenv()

# ==== AWS CONFIG ====
from utils.aws_clients import s3, bedrock, S3_BUCKET, MODEL_NOVA, MODEL_TITAN_EMBED

# ==== STREAMLIT INTERFACE ====
st.title("AI-Powered Recruitment Assistant MVP")
tab1, tab2, tab3, tab4 = st.tabs(["Generate Job", "Upload Resumes", "Match", "Test"])

# --- Job Description Generator ---
with tab1:
    st.header("1. Generate Inclusive Job Description")
    role_input = st.text_input("Enter Role Title", "Data Scientist")
    region_input = st.selectbox(
        'Where will the employee be based?',
        ('Mexico','United States', 'South America'))
    language_input = st.selectbox(
        'In what language should de Job Description be?',
        ('english', 'spanish')
    )

    # Generate and store job description
    if st.button("Generate Description"):
        with st.spinner("Generating..."):
            full_output = generate_job_description(role_input, region_input, language_input)
            validation = validate_job_description(full_output, language_input)
            st.session_state["edited_description"] = full_output
            st.session_state["sections"] = split_job_description_sections(full_output)
            st.session_state["validation"] = validation

    if "sections" in st.session_state:
        for section, content in st.session_state["sections"].items():
            updated = st.text_area(f"✏️ {section}", value=content, height=200)
            st.session_state["sections"][section] = updated  # update state

    if "validation" in st.session_state:
        v = st.session_state["validation"]
        st.markdown("### ✅ Bias & Inclusion Review")
        st.markdown(f"**Result:** `{v.get('result', 'Unknown')}`")
        st.markdown(f"**Reasoning:** {v.get('reasoning', 'No reasoning provided.')}")
        
        if v.get("main_issues_identified"):
            st.markdown("**⚠️ Main Issues Identified:**")
            for issue in v["main_issues_identified"]:
                st.markdown(f"- {issue}")

        if v.get("recommendations"):
            st.markdown("**💡 Recommendations:**")
            for rec in v["recommendations"]:
                st.markdown(f"- {rec}")

    if st.button("Save Job Description"):
        combined = ""
        sections = st.session_state["sections"]
        chunks = get_section_embeddings_dict(sections)

        for section, content in st.session_state["sections"].items():
            combined += f"**{section}**\n\n{content.strip()}\n\n"

        st.session_state["edited_description"] = combined.strip()
        
        save_message = save_job_description_to_json(
            role_input, region_input, language_input,
            chunks=chunks
        )

        st.success(save_message)

    if "sections" in st.session_state and st.button("📄 Generate PDF"):
        with st.spinner("Generating PDF..."):
            html = generate_clean_html(
                st.session_state["sections"],
                role=role_input,
                location=region_input,
                skip_sections = "Average Annual Salary Range in {region}"
            )
            pdf_bytes = generate_pdf_from_html(html)

            st.download_button(
                label="📥 Download",
                data=pdf_bytes,
                file_name=f"JobDescription_{role_input}_{region_input}_.pdf",
                mime="application/pdf"
            )

# --- Resume Upload ---
with tab2:
    st.header("2. Upload Resume(s) for Matching")

    job_map = load_job_descriptions_list()
    if job_map:
        job_options = list(job_map.keys())
        job_labels = [job_map[jid]["label"] for jid in job_options]
        selected_job_id = st.selectbox("Select Job Description", options=job_options, format_func=lambda jid: job_map[jid]["label"])
        selected_job = job_map[selected_job_id]["data"]
    else:
        selected_job_id = None
        selected_job = None
        st.warning("⚠️ No job descriptions available. Please create and save one first.")

    uploaded_files = st.file_uploader("Upload PDF/Docx resume", accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_name = str(uuid.uuid4())
            file_bytes = file.read()

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                tmp.seek(0)
                upload_message = upload_to_s3(tmp, f"resumes/files/{file_name}.pdf")
                st.success(upload_message)

            preview, parsed_text = parse_resume_to_text(file_bytes)
            
            if not parsed_text or len(parsed_text.strip()) < 50:
                st.warning("⚠️ Could not extract meaningful text from the uploaded resume.")
                continue

            lang = detect_language(parsed_text)
            st.markdown(f"**Detected language:** `{lang}`")

            with st.spinner("Extracting sections..."):
                extracted = extract_resume_sections(parsed_text)

            if "error" in extracted:
                st.error(f"Extraction failed: {extracted['error']}")
            else:
                st.session_state["resume_sections"] = extracted

    if "resume_sections" in st.session_state:
        for section, content in st.session_state["resume_sections"].items():
            formatted = format_section_for_display(content)
            updated = st.text_area(f"✏️ {section.title().replace('_', ' ')}", value=formatted, height=200)
            st.session_state["resume_sections"][section] = updated

    if st.button("Save Resume"):
        language = detect_language(parsed_text)
        file_id = file_name

        sections = st.session_state["resume_sections"]

        with st.spinner("Embedding sections..."):
            chunks_dict = embed_resume_sections(sections, bedrock)

        save_message = save_resume_to_json(selected_job_id,language,chunks_dict, file_id)
        st.success(save_message)
    
# --- Resume Vs Job Description Match ---
with tab3:
    st.header("3. Job Descriptions matching")
    job_matching_map = load_job_descriptions_list()
    if job_matching_map:
        job_matching_options = list(job_matching_map.keys())
        job_matching_labels = [job_matching_map[jid]["label"] for jid in job_matching_options]
        job_matching_id = st.selectbox(
            "Select Job Description", 
            options=job_matching_options, 
            format_func=lambda jid: job_matching_map[jid]["label"],
            key = "match_selectbox"
        )
        job_matching = job_matching_map[job_matching_id]["data"]
        job_chunks = get_job_chunks(job_matching_id)
        dict_filtered_resumes = get_matching_resumes(job_matching_id)

        num_resumes = len(dict_filtered_resumes)
        st.markdown(f"**📄 Total resumes found:** `{num_resumes}`")

        if num_resumes > 0:
            top_n = st.number_input(
                "Select number of top resumes to match", 
                min_value=1, 
                max_value=num_resumes, 
                value=min(2, num_resumes),
                step=1
            )

    if st.button("Match"):
        ranking_resumes = rank_resumes_for_job(job_chunks, dict_filtered_resumes, top_n)
        st.session_state["ranking_resumes"] = ranking_resumes
        dict_match_results = {}

        for resume in ranking_resumes:
            ranked_resume_id = resume["resume_id"]
            resume_chunks = dict_filtered_resumes[ranked_resume_id]["chunks"]
            dict_match_results[resume["resume_id"]] = compare_job_resume_embeddings(job_chunks, resume_chunks)

        plot_job_based_radar_multi(dict_match_results)

    else:
        job_matching_id = None
        job_matching = None
        #st.warning("⚠️ No job descriptions available. Please create and save one first.")

    if "ranking_resumes" in st.session_state and st.button("📥 Download Top Matches (Anonymized PDF)"):
        merged_html = "<html><body>"

        for match in st.session_state["ranking_resumes"]:
            resume_id = match["resume_id"]
            resume_data = dict_filtered_resumes.get(resume_id, {})
            resume_chunks = resume_data.get("chunks", {})

            sections_text = {
                k: v["text"]
                for k, v in resume_chunks.items()
                if "text" in v
            }

            resume_html = generate_clean_html(
                sections=sections_text,
                role=f"Candidate #{resume_id[:6]}",
                title="Anonymized Resume",
                skip_sections=["personal_information", "Personal Information"]
            )

            body_start = resume_html.find("<body>")
            body_end = resume_html.find("</body>")
            merged_html += resume_html[body_start + 6:body_end]
            merged_html += "<hr>"

        merged_html += "</body></html>"

        pdf_bytes = generate_pdf_from_html(merged_html)

        st.download_button(
            label="📄 Download Merged Anonymized Resumes",
            data=pdf_bytes,
            file_name="anonymized_resumes.pdf",
            mime="application/pdf"
        )

        
# --- Test Generator ---
with tab4:
    st.header("4. Assessment Generator")
    job_desc = st.text_area("Paste job description to generate a test")
    if st.button("Generate Assessment"):
        with st.spinner("Generating test..."):
            test_output = generate_test_from_description(job_desc)
            st.text_area("Generated Assessment", test_output, height=400)
