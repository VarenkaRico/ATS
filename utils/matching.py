import streamlit as st
import boto3
import json
import os

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

from utils.aws_clients import s3, S3_BUCKET

def load_job_descriptions_list(s3_key="job_descriptions/job_descriptions.json"):
    """
    Load list of job descriptions from S3 for dropdown selection.

    Returns:
        dict: {uuid: {'label': ..., 'data': ...}}
    """
    try:
        s3_object = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        all_jobs = json.loads(s3_object["Body"].read().decode("utf-8"))

        job_map = {}
        for uid, entry in all_jobs.items():
            label = f"{entry['role']} - {entry['region']} ({entry['date_created'][:10]})"
            job_map[uid] = {
                "label": label,
                "data": entry
            }
        return job_map
    except Exception as e:
        st.warning(f"❌ Could not load job descriptions: {e}")
        return {}

def get_resumes_for_job(job_description_id, s3_key = "resumes/resumes.json"):
    """
    Retrieve all resumes that were uploaded for a specific job description ID.

    Args:
        job_description_id (str): The UUID of the job description.
        s3_key (str): Path to the resumes JSON file in S3.

    Returns:
        dict: {resume_id: resume_data, ...} only for resumes matching the job_description_id
    """

    try:
        # Load the full resume database from S3
        s3_obj = s3.get_object(Bucket = S3_BUCKET,
                                      Key = s3_key)
        
        all_resumes = json.loads(s3_obj["Body"].read().decode("utf-8"))

        #Filter resumes by job description ID
        filtered = {
            resume_id: data
            for resume_id, data in all_resumes.items()
                if data.get("role") == job_description_id
        }
        
        return filtered
    
    except s3.exceptions.NoSuchKey:
        print("No resume file found.")
        return {}
    
    except Exception as e:
        print(f"Error loading resumes: {e}")
        return {}

def load_chunks_from_s3(object_id, type):
    if type == "resume":
        s3_key = "resumes/resumes.json"
    elif type == "job description":
        s3_key = "job_descriptions/job_descriptions.json"

    try:
        s3_object = s3.get_object(Bucket = S3_BUCKET, Key = s3_key)
        job_data = json.loads(s3_object["Body"].read().decode("utf-8"))
        if object_id in job_data:
            return job_data[object_id]["chunks"]
        
        else:
            raise ValueError(f"{type} ID '{object_id}' not found in {s3_key}")
    
    except Exception as e:
        #st.error(f"❌ Error loading job chunks: {e}")
        print(f"❌ Error loading job chunks: {e}")
        return []
    
def get_info_for_job(job_description_id, type):
    """
    Retrieve all resumes that were uploaded for a specific job description ID.

    Args:
        job_description_id (str): The UUID of the job description.
        s3_key (str): Path to the resumes JSON file in S3.

    Returns:
        dict: {resume_id: resume_data, ...} only for resumes matching the job_description_id
    """

    if type == "resumes":
        s3_key ="resumes/resumes.json"

    elif type == "job description":
        s3_key = "job_descriptions/job_descriptions.json"

    try:
        # Load the full resume database from S3
        s3_obj = s3.get_object(Bucket = S3_BUCKET,
                                      Key = s3_key)
        
        all_info = json.loads(s3_obj["Body"].read().decode("utf-8"))

        #Filter resumes by job description ID

        if type == "resumes":
            filtered = {
                resume_id: data
                for resume_id, data in all_info.items()
                    if data.get("role") == job_description_id
            }

        elif type == "job description":
            filtered = all_info[job_description_id]
        
        return filtered
    
    except s3.exceptions.NoSuchKey:
        print("No resume file found.")
        return {}
    
    except Exception as e:
        print(f"Error loading resumes: {e}")
        return {}

def compute_gap_boost_score(job_chunk, resume_chunks, required_sections, similarity_threshold = 0, max_boost = 0.25):
    """
    Compute a boost score by comparing non-required resume sections to job-resume embedding gap.

        Args:
            job_chunk (dict): {section: {text, embedding}}
            resume_chunks (dict): {section: {text. embedding}}
            required_sections (tuple): Resume sections considered as core
            similarity_threshold (float): Minimum similarity to count as meaningful
            max_boost (float): Maximum value boost can contribute

        Returns:
            dict with:
                boost: float
                aligned_sections: list of resume section names contributing
                gap_vector: np.array (for debug)
    """

    job_vecs = [np.array(c["embedding"]) for c in job_chunk.values() if "embedding" in c]
    req_vecs = [np.array(resume_chunks[s]["embedding"]) for s in required_sections if s in resume_chunks]

    if not job_vecs or not req_vecs:
        return {"boost": 0.0,
                "aligned_sections": [],
                "gap_vector": None}
    
    job_mean = np.mean(job_vecs, axis = 0)
    req_mean = np.mean(req_vecs, axis = 0)
    gap_vec = job_mean - req_mean
    gap_vec = gap_vec.reshape(1, -1)

    boost_sections = []
    sim_scores = []

    for sec, data in resume_chunks.items():
        if sec in required_sections or "embedding" not in data:
            continue
        res_vec = np.array(data["embedding"]).reshape(1, -1)
        sim = cosine_similarity(gap_vec, res_vec)[0][0]

        if sim > similarity_threshold:
            boost_sections.append(sec)
            sim_scores.append(sim)

    if sim_scores:
        avg_sim = np.mean(sim_scores)
        boost = min(max_boost, avg_sim * max_boost)

    else:
        boost = 0.0

    return {
        "boost": round(boost, 3),
        "aligned_sections": boost_sections,
        "gap_vector": gap_vec
    }

def compute_resume_score_with_required_sections(
        section_scores,
        job_chunk, 
        resume_chunks,
        required_sections = ("skills", "experience", "education_studies"),
        optional_boost = True
):
    """
    Hybrid score combining required-section emphasis and optional boosts.

    Args:
        section_scores (dict): {section_name: similarity_score}
        required_sections (tuple): Key sections needed for a complete match
        optional_boost (bool): Whether to let other sections boost the score

    Returns:
        float: Final score between 0.0 and 1.0
    """

    required_scores = [section_scores[sec] for sec in required_sections if sec in section_scores]
    optional_scores = [score for sec, score in section_scores.items() if sec not in required_scores]

    #Compute base average from required sections
    if required_scores:
        base_score = np.mean(required_scores)

    else:
        base_score = 0.0

    # Optional section boost (e.g. Projects, Certifications)
    boost = 0.0

    if optional_boost and optional_scores:
        dict_boost = compute_gap_boost_score(job_chunk, resume_chunks, required_sections)
        boost = dict_boost["boost"]

    #Final score: average of required + optional boost

    final_score = base_score * (1 + boost)
    
    return final_score,dict_boost

def get_job_chunks(job_description_id):
    try:
        # Load the full resume database from S3
        s3_obj = s3.get_object(Bucket = S3_BUCKET,
                                      Key = "job_descriptions/job_descriptions.json")
        
        all_job_descriptions = json.loads(s3_obj["Body"].read().decode("utf-8"))

        job_data = all_job_descriptions[job_description_id]

        return job_data.get("chunks", {})

    except Exception as e:
        print("No job description with the id '{job_description_id}' was found")
        return {}
    
def get_matching_resumes(job_description_id):
    try:
        # Load the full resume database from S3
        s3_obj = s3.get_object(Bucket = S3_BUCKET,
                                      Key = "resumes/resumes.json")
        
        all_resumes = json.loads(s3_obj["Body"].read().decode("utf-8"))

        dict_filtered_resumes = {
            resume_id: data
                for resume_id, data in all_resumes.items()
                    if data.get("role") == job_description_id
        }

        return dict_filtered_resumes
    except Exception as e:
        print("No resumes for the job desription with the id '{job_descripton_id}' were found")
        return {}

def rank_resumes_for_job(job_chunks, dict_matching_resumes, top_n=5):
    """
    Compare all resumes linked to a specific job description and return the top N ranked matches.

    Args:
        job_description_id (str): UUID of the selected job description.
        job_descriptions_dict (dict): All job descriptions loaded from S3.
        resumes_dict (dict): All resumes loaded from S3.
        top_n (int): Number of top matches to return.

    Returns:
        list of dicts: Each with resume_id, score, file name, and matched sections.
    """

    job_chunks.pop('Average Annual Salary Range in Mexico (USD)')
    job_chunks.pop('Equal Opportunity Statement')
    job_chunks.pop('Application Process')


    job_vectors = [
        np.array(chunk["embedding"]).reshape(1, -1)
        for chunk in job_chunks.values()
        if "embedding" in chunk
    ]

    if not job_vectors:
        raise ValueError("Job description has no valid embeddings.")

    job_matrix = np.vstack(job_vectors)

    # 2. Filter resumes for the specific job

    results = []

    # 3. Compare each resume
    for resume_id, resume_data in dict_matching_resumes.items():
        resume_chunks = resume_data.get("chunks", {})
        resume_chunks.pop('personal_information')
        section_scores = {}
        sim_scores = []

        for res_sec, rc in resume_chunks.items():
            
            if "embedding" not in rc:
                continue
            vec = np.array(rc["embedding"]).reshape(1, -1)
            sim = cosine_similarity(vec, job_matrix)
            best_sim = float(np.max(sim))
            sim_scores.append(best_sim)
            section_scores[res_sec] = round(best_sim, 3)

        avg_score, dict_boost = compute_resume_score_with_required_sections(section_scores, job_chunks, resume_chunks)

        results.append({
            "resume_id": resume_id,
            "file": resume_data.get("file"),
            "score": avg_score,
            "section_scores": section_scores,
            "boost": dict_boost
        })

    # 4. Sort by score and return top N
    return sorted(results, key=lambda r: r["score"], reverse=True)[:top_n]

def compare_job_resume_embeddings(job_chunks: dict, resume_chunks: dict):
    """
    Compare embeddings between job description sections and resume sections.

    Args:
        job_chunks (dict): {"section_name": {"text": ..., "embedding": [...]}, ...}
        resume_chunks (dict): same structure

    Returns:
        dict: {
            "score": overall_score (float),
            "section_matches": {
                "Job Description": {
                    "best_resume_section": "general_information",
                    "similarity": 0.89
                },
                ...
            }
        }
    """
    results = []
    section_matches = {}

    for job_sec, job_data in job_chunks.items():
        job_vec = np.array(job_data["embedding"]).reshape(1, -1)
        best_score = 0
        best_match = None

        for res_sec, res_data in resume_chunks.items():
            res_vec = np.array(res_data["embedding"]).reshape(1, -1)
            sim = cosine_similarity(job_vec, res_vec)[0][0]

            if sim > best_score:
                best_score = sim
                best_match = res_sec

        section_matches[job_sec] = {
            "best_resume_section": best_match,
            "similarity": round(best_score, 3)
        }
        results.append(best_score)

    overall_score = round(float(np.mean(results)), 3) if results else 0.0

    return {
        "score": overall_score,
        "section_matches": section_matches
    }

def plot_job_based_radar_multi(match_results_dict, labels_by="Resume"):
    """
    Plot a radar chart comparing multiple resumes' match scores per job section.

    Args:
        match_results_dict (dict): {
            "resume_name_or_id": {
                "section_matches": {
                    "Job Description": {"similarity": 0.87, "best_resume_section": "..."},
                    ...
                }
            },
            ...
        }
        labels_by (str): 'Resume' or 'Job Section' for label style
    """
    # Extract all job sections from first entry
    first_key = next(iter(match_results_dict))
    job_sections = list(match_results_dict[first_key]["section_matches"].keys())

    # Complete the loop
    labels = job_sections
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = ["#2196F3", "#4CAF50", "#FF5722"]  # Add more if needed
    for i, (resume_id, result) in enumerate(match_results_dict.items()):
        matches = result["section_matches"]
        scores = [matches[sec]["similarity"] for sec in job_sections]
        labels += [labels[0]]
        angles += [angles[0]]
        scores.append(scores[0])  # close the loop

        ax.plot(angles, scores, label=resume_id, linewidth=2, color=colors[i % len(colors)])
        ax.fill(angles, scores, alpha=0.15, color=colors[i % len(colors)])

    ax.set_title("Similarity by Job Description Section", size=14, pad=20)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), title="Resumes")

    st.pyplot(fig)