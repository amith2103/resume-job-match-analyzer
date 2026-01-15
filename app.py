import os
import re
import json
import sqlite3
from datetime import datetime
from io import BytesIO

import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document

from skills import SKILLS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Optional PDF export (ReportLab)
# =========================
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# =========================
# Page Config
# =========================
st.set_page_config(page_title="Resume Job Match Analyzer", layout="wide")


# =========================
# CSS (compact top + tags)
# =========================
st.markdown(
    """
    <style>
        header[data-testid="stHeader"] {display: none;}
        .block-container {padding-top: 0.2rem !important; padding-bottom: 1rem;}
        h1, h2, h3 {margin-top: 0.4rem !important; margin-bottom: 0.4rem !important;}

        .tag {
            display:inline-block;
            padding:6px 10px;
            margin:4px 6px 4px 0px;
            border-radius:999px;
            border:1px solid rgba(0,0,0,0.10);
            font-size:14px;
            background: #F7F7FF;
        }
        .tag-green { background: #E9FBF1; border-color:#BDE8CF; }
        .tag-red   { background: #FFECEC; border-color:#FFC3C3; }
        .tag-blue  { background: #EAF2FF; border-color:#BED3FF; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Header
# =========================
st.markdown("## Resume ↔ Job Match Analyzer")
st.caption("Upload a resume (PDF/DOCX), paste a job description, and get skill match + NLP similarity.")
st.divider()


# =========================
# Step 5: SQLite (absolute path)
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "analysis_history.db")


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            resume_file_name TEXT,
            job_title TEXT,
            company TEXT,
            skill_match REAL,
            text_similarity REAL,
            resume_skills TEXT,
            job_skills TEXT,
            matched_skills TEXT,
            missing_skills TEXT,
            detected_job_skills TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def ensure_column_detected_job_skills():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(analysis_results);")
    existing_cols = {row[1] for row in cur.fetchall()}

    if "detected_job_skills" not in existing_cols:
        cur.execute("ALTER TABLE analysis_results ADD COLUMN detected_job_skills TEXT;")
        conn.commit()

    conn.close()


def save_result(
    resume_file_name,
    job_title,
    company,
    skill_match,
    text_similarity,
    resume_skills,
    job_skills,
    matched_skills,
    missing_skills,
    detected_job_skills
):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analysis_results
        (created_at, resume_file_name, job_title, company, skill_match, text_similarity,
         resume_skills, job_skills, matched_skills, missing_skills, detected_job_skills)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            resume_file_name,
            job_title,
            company,
            skill_match,
            text_similarity,
            json.dumps(resume_skills),
            json.dumps(job_skills),
            json.dumps(matched_skills),
            json.dumps(missing_skills),
            json.dumps(detected_job_skills),
        ),
    )
    conn.commit()
    conn.close()


def load_history():
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            id,
            created_at,
            resume_file_name,
            job_title,
            company,
            skill_match,
            text_similarity,
            matched_skills,
            missing_skills,
            detected_job_skills
        FROM analysis_results
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()

    if not df.empty:
        df["matched_skills"] = df["matched_skills"].apply(lambda x: ", ".join(json.loads(x)) if x else "")
        df["missing_skills"] = df["missing_skills"].apply(lambda x: ", ".join(json.loads(x)) if x else "")
        df["detected_job_skills"] = df["detected_job_skills"].apply(lambda x: ", ".join(json.loads(x)) if x else "")
    return df


def clear_history():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM analysis_results;")
    conn.commit()
    conn.close()


init_db()
ensure_column_detected_job_skills()


# =========================
# Step 7 helpers: reports
# =========================
def build_analysis_dataframe(job_title, company, skill_match, text_similarity,
                             detected_job_skills, matched_skills, missing_skills):
    rows = [
        ["Job Title", job_title or ""],
        ["Company", company or ""],
        ["Skill Match (%)", "" if skill_match is None else skill_match],
        ["Text Similarity (%)", text_similarity],
        ["Auto-detected JD Skills", ", ".join(detected_job_skills)],
        ["Matched Skills", ", ".join(matched_skills)],
        ["Missing Skills", ", ".join(missing_skills)],
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def build_text_report(job_title, company, skill_match, text_similarity,
                      detected_job_skills, matched_skills, missing_skills):
    return f"""Resume ↔ Job Match Report
========================

Job Title: {job_title or ""}
Company: {company or ""}
Skill Match (%): {"" if skill_match is None else skill_match}
Text Similarity (%): {text_similarity}

Auto-detected Job Skills:
- {", ".join(detected_job_skills) if detected_job_skills else "None"}

Matched Skills:
- {", ".join(matched_skills) if matched_skills else "None"}

Missing Skills (Skill Gap):
- {", ".join(missing_skills) if missing_skills else "None"}

Tips:
- If similarity is low, rewrite resume bullets using keywords from the job description.
- If missing skills exist, add those skills to resume only if you truly have them.
"""


def build_pdf_report_bytes(report_text: str) -> bytes:
    """
    Builds a PDF in memory.
    Only works if reportlab is installed.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not installed")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    x = 50
    y = height - 60
    line_height = 14

    for line in report_text.splitlines():
        if y < 60:
            c.showPage()
            y = height - 60
        c.drawString(x, y, line[:110])  # prevent long line overflow
        y -= line_height

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# =========================
# File text extraction
# =========================
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(docx_file):
    document = Document(docx_file)
    text = ""
    for paragraph in document.paragraphs:
        if paragraph.text:
            text += paragraph.text + "\n"
    return text


# =========================
# Step 6: Auto skill detection (tech only)
# =========================
STOPWORDS_BASIC = {
    "the","and","or","to","of","in","for","with","on","a","an","as","is","are","be","by","from","at",
    "this","that","it","we","you","your","our","they","their","will","can","may","must","should",
    "experience","years","year","strong","excellent","skills","skill","knowledge","ability","required",
    "preferred","responsibilities","requirements","role","work","working","team","teams","using",
    "including","etc","based","basic","being","both","also","allow","additional","addition",
    "company","business","clients","audience","benefits","best","between","broad","common","communicate",
    "assess","applicants","applicable","alternatively","alternate","advertising","color","complex",
    "comprehensive","accuracy","analyses","analytics","analyze","apply","bachelor","bonus","billion"
}

TECH_KEYWORDS = {
    # languages
    "python","java","javascript","typescript","sql","r","c","c++","c#","go","rust","php","scala","kotlin","swift",
    # frameworks / libraries
    "react","angular","vue","next.js","node.js","express","django","flask","fastapi",
    "spring","spring boot",".net","asp.net","bootstrap","tailwind","pandas","numpy","scikit-learn","tensorflow","pytorch",
    # cloud / devops / data
    "aws","azure","gcp","docker","kubernetes","git","github","gitlab","ci/cd","jenkins","airflow","spark","databricks",
    # databases
    "mysql","postgresql","postgres","mongodb","sql server","snowflake","redshift","bigquery",
    # BI / analytics
    "power bi","tableau","excel","looker"
}

ABBREV_MAP = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "powerbi": "power bi",
    "ms sql": "sql server",
    "mssql": "sql server",
    "node": "node.js",
    "nodejs": "node.js",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "dotnet": ".net",
}

def normalize_term(t: str) -> str:
    t = t.strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t)

    t = t.replace("c sharp", "c#")
    t = t.replace("c plus plus", "c++")
    t = t.replace("reactjs", "react")
    t = t.replace("react.js", "react")
    t = t.replace("node js", "node.js")
    t = t.replace("nodejs", "node.js")

    if t in ABBREV_MAP:
        t = ABBREV_MAP[t]

    return t


def detect_job_skills(job_text: str, top_k: int = 30) -> list[str]:
    """
    Tech-only skill detection:
    - catches .net, c#, c++, node.js, power bi, SQL server, etc.
    - uses TF-IDF but filters to tech terms only
    """
    text = job_text.strip()
    if not text:
        return []

    t = text.lower()
    special_hits = set()

    if re.search(r"\b\.net\b", t) or re.search(r"\bdotnet\b", t):
        special_hits.add(".net")
    if re.search(r"\bc\#\b", text, flags=re.I):
        special_hits.add("c#")
    if re.search(r"\bc\+\+\b", text, flags=re.I):
        special_hits.add("c++")
    if re.search(r"\bnode\.js\b", t) or re.search(r"\bnodejs\b", t):
        special_hits.add("node.js")
    if re.search(r"\bpower\s*bi\b", t) or re.search(r"\bpowerbi\b", t):
        special_hits.add("power bi")
    if re.search(r"\bsql\s*server\b", t) or re.search(r"\bmssql\b", t):
        special_hits.add("sql server")

    raw_words = re.findall(r"[a-z0-9\+\#\.]+", t)
    token_hits = set()
    for w in raw_words:
        w = normalize_term(w)
        if w in STOPWORDS_BASIC:
            continue
        if w in TECH_KEYWORDS:
            token_hits.add(w)

    tfidf_hits = set()
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=300)
        X = vec.fit_transform([t])
        names = vec.get_feature_names_out()
        scores = X.toarray()[0]
        top_idx = scores.argsort()[::-1][:top_k]

        for i in top_idx:
            term = normalize_term(names[i])
            if term in STOPWORDS_BASIC:
                continue
            if term in TECH_KEYWORDS or any(sym in term for sym in [".", "+", "#"]):
                tfidf_hits.add(term)
    except Exception:
        pass

    detected = sorted(set(special_hits | token_hits | tfidf_hits))
    return detected


# =========================
# Skill extraction + display
# =========================
def extract_skills(text, skills_list):
    text = text.lower()
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return sorted(list(set(found)))


def show_skill_tags(title, skills, kind="green"):
    st.markdown(f"### {title}")
    if len(skills) == 0:
        st.info("Nothing to show here.")
        return

    cls = "tag-green" if kind == "green" else ("tag-red" if kind == "red" else "tag-blue")
    tags_html = ""
    for s in skills:
        tags_html += f'<span class="tag {cls}">{s}</span>'
    st.markdown(tags_html, unsafe_allow_html=True)


# =========================
# Text similarity
# =========================
def compute_text_similarity(resume_text, job_text):
    docs = [resume_text, job_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


# =========================
# Layout
# =========================
left_col, right_col = st.columns([1.05, 1])

with left_col:
    st.subheader("Inputs")
    st.caption("Tip: paste a full job description for better similarity score.")

    job_title = st.text_input("Job Title (optional)", placeholder="e.g., Junior Data Analyst")
    company = st.text_input("Company (optional)", placeholder="e.g., T-Mobile")

    uploaded_file = st.file_uploader("Upload resume (PDF or Word)", type=["pdf", "docx"])

    if "resume_text" not in st.session_state:
        st.session_state["resume_text"] = ""
    if "resume_file_name" not in st.session_state:
        st.session_state["resume_file_name"] = ""

    if uploaded_file is not None:
        st.session_state["resume_file_name"] = uploaded_file.name
        file_name = uploaded_file.name.lower()
        with st.spinner("Extracting resume text..."):
            if file_name.endswith(".pdf"):
                st.session_state["resume_text"] = extract_text_from_pdf(uploaded_file)
            elif file_name.endswith(".docx"):
                st.session_state["resume_text"] = extract_text_from_docx(uploaded_file)
        st.success("Resume uploaded and text extracted!")

    st.session_state["resume_text"] = st.text_area(
        "Resume Text",
        value=st.session_state["resume_text"],
        height=280
    )

    job_text = st.text_area("Job Description", height=280)
    analyze_clicked = st.button("✨ Analyze Match", type="primary", use_container_width=True)


with right_col:
    tab_results, tab_history = st.tabs(["Results", "History"])

    with tab_results:
        st.subheader("Results")
        st.caption("Your scores and gaps will appear here.")

        if analyze_clicked:
            resume_text = st.session_state["resume_text"]
            resume_file_name = st.session_state.get("resume_file_name", "")

            if resume_text.strip() == "" or job_text.strip() == "":
                st.warning("Please provide both Resume text and Job Description.")
            else:
                prog = st.progress(0)
                prog.progress(15)

                detected_job_skills = detect_job_skills(job_text)
                combined_skills = sorted(set([normalize_term(s) for s in SKILLS]) | set(detected_job_skills))

                prog.progress(35)

                resume_skills = extract_skills(resume_text, combined_skills)
                job_skills = extract_skills(job_text, combined_skills)

                prog.progress(60)

                matched_skills = sorted(list(set(resume_skills) & set(job_skills)))
                missing_skills = sorted(list(set(job_skills) - set(resume_skills)))

                prog.progress(75)

                similarity = compute_text_similarity(resume_text, job_text)
                similarity_percent = round(similarity * 100, 2)

                if len(job_skills) == 0:
                    skill_match_score = None
                else:
                    skill_match_score = round((len(matched_skills) / len(job_skills)) * 100, 2)

                prog.progress(100)
                st.write("")

                # Save to DB
                save_result(
                    resume_file_name=resume_file_name,
                    job_title=job_title.strip() if job_title else None,
                    company=company.strip() if company else None,
                    skill_match=skill_match_score,
                    text_similarity=similarity_percent,
                    resume_skills=resume_skills,
                    job_skills=job_skills,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    detected_job_skills=detected_job_skills
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Skill Match (%)", "N/A" if skill_match_score is None else skill_match_score)
                    if skill_match_score is None:
                        st.caption("No job skills detected (try a longer JD).")
                with c2:
                    st.metric("Text Similarity (%)", similarity_percent)
                    st.caption("TF-IDF + cosine similarity (0–100).")

                st.divider()

                show_skill_tags("Auto-detected Skills from Job Description", detected_job_skills, kind="blue")
                st.divider()

                show_skill_tags("Matched Skills", matched_skills, kind="green")

                if len(job_skills) > 0 and len(missing_skills) == 0:
                    st.success("Perfect! You already have every skill mentioned in the job description ✅")
                elif len(job_skills) > 0:
                    show_skill_tags("Missing Skills (Skill Gap)", missing_skills, kind="red")
                else:
                    st.info("Tip: Paste a longer JD to detect more skills.")

                st.divider()

                st.markdown("### Skill Gap Summary Chart")
                chart_df = pd.DataFrame({
                    "Category": ["Matched Skills", "Missing Skills"],
                    "Count": [len(matched_skills), len(missing_skills)]
                })
                fig, ax = plt.subplots()
                ax.bar(chart_df["Category"], chart_df["Count"])
                ax.set_ylabel("Count")
                st.pyplot(fig)

                with st.expander("See extracted skills (Resume vs Job)"):
                    show_skill_tags("Resume Skills", resume_skills, kind="green")
                    show_skill_tags("Job Skills", job_skills, kind="blue")

                # ===============================
                # STEP 7: DOWNLOAD REPORT (SAFE)
                # ===============================
                st.divider()
                st.subheader("Download Report")

                analysis_df = build_analysis_dataframe(
                    job_title=job_title,
                    company=company,
                    skill_match=skill_match_score,
                    text_similarity=similarity_percent,
                    detected_job_skills=detected_job_skills,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills
                )

                txt_report = build_text_report(
                    job_title=job_title,
                    company=company,
                    skill_match=skill_match_score,
                    text_similarity=similarity_percent,
                    detected_job_skills=detected_job_skills,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills
                )

                dc1, dc2, dc3 = st.columns(3)

                with dc1:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=analysis_df.to_csv(index=False).encode("utf-8"),
                        file_name="resume_job_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with dc2:
                    st.download_button(
                        "⬇️ Download TXT",
                        data=txt_report.encode("utf-8"),
                        file_name="resume_job_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                with dc3:
                    if REPORTLAB_AVAILABLE:
                        pdf_bytes = build_pdf_report_bytes(txt_report)
                        st.download_button(
                            "⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name="resume_job_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.caption("PDF export disabled (install: pip install reportlab)")

        else:
            st.info("Click “✨ Analyze Match” to see results.")

    with tab_history:
        st.subheader("History")
        st.caption("All your past analyses are stored in SQLite and shown here.")

        hist_df = load_history()
        st.dataframe(hist_df, use_container_width=True)

        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_bytes,
            file_name="analysis_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history()
            st.success("History cleared!")
            st.rerun()
