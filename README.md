# 📄 Resume ↔ Job Match Analyzer

## Smart Resume–Job Matching & Skill Gap Analysis Web Application

👋 Welcome to Resume ↔ Job Match Analyzer!

Resume ↔ Job Match Analyzer is a modern, NLP-powered web application designed to help job seekers understand how well their resume matches a job description — and what skills they are missing — using data-driven insights.
This project was built using Vibe Coding, where features were designed in plain English and iteratively refined with AI assistance.
The goal was to demonstrate how real-world, production-style NLP applications can be built quickly while maintaining clean code, explainability, and a professional UI.

🎯 What Does This App Do?

📄 Upload your resume (PDF or DOCX)

📝 Paste a job description

📊 Instantly see how well you match the role

Key Insights Provided:

✅ Skill Match Percentage

📈 NLP-based Text Similarity Score

🧩 Matched Skills

❌ Missing Skills (Skill Gap)

📊 Visual Skill Gap Charts

🗂️ History of all past analyses

---

## ▶️ Why This Project?

Job descriptions are often written in ATS-optimized language, and resumes fail not because of lack of skill — but because of keyword mismatch.

I built this project to:

- Help candidates optimize resumes intelligently
- Demonstrate practical NLP usage (not toy examples)
- Build a portfolio-ready full-stack Python app
- Show how AI-assisted development can speed up real-world projects
- This is not just a demo — it solves a real hiring problem.

---

## 🚀 Core Features
🔹 Resume Upload & Parsing

▶️Upload resume as PDF or DOCX

▶️Automatic text extraction

▶️Editable resume text area

🔹 Smart Job Skill Detection

▶️Auto-detects technical skills from job descriptions

▶️Handles real-world terms like:

▶️.NET, C++, C#, Node.js, Power BI

▶️Normalizes abbreviations (e.g., ML → Machine Learning)

🔹 Skill Match Analysis

▶️Matches resume skills against job skills

Displays:

▶️Matched Skills

▶️Missing Skills (Skill Gap)

▶️Shows a perfect match message when no skills are missing

🔹 NLP Text Similarity

▶️Uses TF-IDF Vectorization

▶️Computes Cosine Similarity

▶️Measures semantic alignment between resume & job description

🔹 Visual Insights

▶️Skill Gap bar chart

▶️Tag-based skill visualization

▶️Clean side-by-side UI

🔹 Analysis History

▶️Stores all analyses in SQLite

▶️View past results anytime

▶️Export full history as CSV

🔹 Export Reports

Download results as:

📄 CSV

📝 TXT

📕 PDF (optional)

---



## 🛠️ Tech Stack
### Frontend / UI

➡️ Streamlit

➡️ Custom CSS styling

➡️ Responsive column-based layout

### NLP & Data Processing

➡️ Python

➡️ scikit-learn (TF-IDF, Cosine Similarity)

➡️ pandas

➡️ Regular Expressions

### Visualization

➡️ matplotlib

### File Handling

➡️ pdfplumber (PDF parsing)

➡️ python-docx (Word parsing)

### Database

➡️ SQLite (local persistence)

---

## 🗂️ Project Structure

resume-job-match-analyzer

├── app.py                  # Main Streamlit application

├── skills.py               # Curated base skills list

├── requirements.txt        # Python dependencies

├── README.md               # Project documentation

├── .gitignore              # Ignored files & folders

├── analysis_history.db     # SQLite database (local)

└── __pycache__/            # Python cache (ignored)

---

## 📊 Metrics Explained
 Metric	                              Description

Skill Match (%):-                	% of job skills found in resume

Text Similarity :-               (%)	NLP similarity using TF-IDF + cosine similarity

Matched Skills	:-               Skills present in both resume & job

Missing Skills :-	               Skills required by job but missing in resume

---

## ⚙️ How It Works (High-Level Flow)

1️⃣ Upload resume or paste resume text

2️⃣ Paste job description

3️⃣ Auto-detect job skills using NLP

4️⃣ Extract resume & job skills

5️⃣ Compute skill match percentage

6️⃣ Compute text similarity score

7️⃣ Visualize skill gaps

8️⃣ Save analysis to database

9️⃣ Export results

---

## 🚀 Running the Project Locally

git clone https://github.com/amith2103/resume-job-match-analyzer.git

cd resume-job-match-analyzer

pip install -r requirements.txt

streamlit run app.py

---

## 🌱 Future Enhancements

▪️ATS keyword weighting

▪️Resume rewrite suggestions

▪️Role-based skill templates

▪️Cloud database integration

▪️Authentication & user accounts

▪️Deployment on Streamlit Cloud

