import streamlit as st
import fitz
import re
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

from transformers import pipeline

from sentence_transformers import (
    SentenceTransformer,
    util
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

# ======================================================
# 🎯 PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="    ",
    layout="wide"
)

st.title("🧭 Governance Framework for Agentic AI in Fintech")

st.markdown("""
AI-Driven Multi-Agent Governance System

✔ FMR Compliance Agent
✔ Semantic AML Agent
✔ RAG + FAISS Retrieval
✔ Explainable AI
✔ PDF Agreement Generation
✔ AML Highlight Report
""")

VECTOR_PATH = r"C:\Users\Asif\Desktop\Thesis_GM\fund_policy_index_hf"

# ======================================================
# 🔥 LOAD MODELS
# ======================================================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_classifier():

    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

@st.cache_resource
def load_aml_semantic_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

vectorstore = load_vectorstore()

classifier = load_classifier()

semantic_model = load_aml_semantic_model()

labels = [
    "Compliant",
    "Partially Compliant",
    "Non-Compliant"
]

# ======================================================
# 🧠 SESSION STATE
# ======================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "reports" not in st.session_state:
    st.session_state.reports = {}

# ======================================================
# 🧹 CLEAN TEXT
# ======================================================

def clean(text):

    return re.sub(
        r"\s+",
        " ",
        text
    )[:250]

# ======================================================
# 🚨 DOCUMENT VALIDATOR
# ======================================================

def validate_document(text):

    text = text.lower()

    finance_keywords = [

        "fund",
        "risk",
        "investment",
        "liquidity",
        "financial",
        "bank",
        "compliance",
        "audit",
        "governance",
        "portfolio",
        "market risk",
        "policy"

    ]

    score = 0

    for k in finance_keywords:

        if k in text:
            score += 1

    return score >= 3

# ======================================================
# 🧠 FMR AGENT
# ======================================================

def fmr_agent(results):

    policies = []

    results = results[:10]

    texts = [
        r.page_content[:256]
        for r in results
    ]

    outputs = classifier(
        texts,
        candidate_labels=labels
    )

    for r, out in zip(results, outputs):

        label = out["labels"][0]

        confidence = round(
            out["scores"][0],
            2
        )

        score = (
            1.0
            if label == "Compliant"
            else 0.5
            if label == "Partially Compliant"
            else 0.0
        )

        policies.append({

            "policy":
                r.metadata.get(
                    "source",
                    "Unknown"
                ),

            "status":
                label,

            "confidence":
                confidence,

            "score":
                score,

            "evidence":
                clean(r.page_content)

        })

    return policies

# ======================================================
# 📊 ANALYSIS ENGINE
# ======================================================

def analyze(policies):

    compliant = [
        p for p in policies
        if p["status"] == "Compliant"
    ]

    partial = [
        p for p in policies
        if p["status"] == "Partially Compliant"
    ]

    non = [
        p for p in policies
        if p["status"] == "Non-Compliant"
    ]

    overall_score = (
        sum(p["score"] for p in policies)
        / len(policies)
    )

    return {

        "compliant": compliant,

        "partial": partial,

        "non": non,

        "score": overall_score
    }

# ======================================================
# 🧠 SEMANTIC AML AGENT
# ======================================================

def aml_agent(text):

    suspicious_patterns = [

        "money laundering through offshore accounts",

        "suspicious shell company transactions",

        "terrorist financing activity",

        "unusual large cash movement",

        "fraudulent financial activity",

        "cross border hidden transfers",

        "unverified third party transactions",

        "illegal financial structuring",

        "anonymous high value transactions"
    ]

    sentences = re.split(
        r'(?<=[.!?]) +',
        text
    )

    suspicious_sentences = []

    total_score = 0

    for sentence in sentences:

        if len(sentence.strip()) < 20:
            continue

        sentence_embedding = semantic_model.encode(
            sentence,
            convert_to_tensor=True
        )

        similarities = []

        for pattern in suspicious_patterns:

            pattern_embedding = semantic_model.encode(
                pattern,
                convert_to_tensor=True
            )

            similarity = util.cos_sim(
                sentence_embedding,
                pattern_embedding
            ).item()

            similarities.append(similarity)

        max_similarity = max(similarities)

        if max_similarity > 0.55:

            suspicious_sentences.append({

                "sentence":
                    sentence,

                "score":
                    round(max_similarity, 2)

            })

            total_score += max_similarity * 100

    total_score = min(total_score, 100)

    if total_score >= 70:

        risk = "High Risk"

    elif total_score >= 35:

        risk = "Medium Risk"

    else:

        risk = "Low Risk"

    if suspicious_sentences:

        reason = (
            f"{len(suspicious_sentences)} "
            f"suspicious financial "
            f"sentence(s) detected."
        )

    else:

        reason = (
            "No semantically suspicious "
            "AML activity detected."
        )

    return {

        "risk": risk,

        "score": round(total_score, 2),

        "reason": reason,

        "sentences": suspicious_sentences
    }

# ======================================================
# 📊 PIE CHART
# ======================================================

def chart(data):

    fig, ax = plt.subplots()

    ax.pie(

        [
            len(data["compliant"]),
            len(data["partial"]),
            len(data["non"])
        ],

        labels=[
            "Compliant",
            "Partial",
            "Non-Compliant"
        ],

        autopct="%1.1f%%"
    )

    st.pyplot(fig)

# ======================================================
# 🧾 AGREEMENT GENERATOR
# ======================================================

def agreement(
    analysis,
    aml,
    level,
    decision
):

    def fmt(items):

        if not items:
            return "None"

        return "\n".join(
            [f"- {p['policy']}" for p in items]
        )

    compliant_list = fmt(
        analysis["compliant"]
    )

    partial_list = fmt(
        analysis["partial"]
    )

    non_list = fmt(
        analysis["non"]
    )

    return f"""
FINANCIAL COMPLIANCE AGREEMENT

Date:
{datetime.now().strftime("%Y-%m-%d %H:%M")}

====================================================

1. EXECUTIVE SUMMARY

Compliance Level:
{level}

AML Risk:
{aml['risk']}

====================================================

2. COMPLIANCE SCORE

Overall Score:
{round(analysis['score'],2)}

====================================================

3. AML RISK ANALYSIS

{aml['reason']}

====================================================

4. COMPLIANT POLICIES

{compliant_list}

====================================================

5. PARTIALLY COMPLIANT POLICIES

{partial_list}

====================================================

6. NON-COMPLIANT POLICIES

{non_list}

====================================================

7. FINAL DECISION

{decision}

====================================================

8. EXPLAINABILITY

Decision generated using:

- RAG Retrieval System
- FMR Compliance Agent
- Semantic AML AI Agent
- AI Governance Engine

====================================================

Authorized By:

AI Governance Compliance System
"""

# ======================================================
# 📄 PDF GENERATOR
# ======================================================

def generate_pdf(text):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    for line in text.split("\n"):

        if line.strip():

            story.append(
                Paragraph(
                    line,
                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(1, 8)
            )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "---- End of Report ----",
            styles["Italic"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer

# ======================================================
# 🚨 AML HIGHLIGHT REPORT
# ======================================================

def generate_aml_report(aml_data):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AML Suspicious Activity Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            f"AML Risk Level: {aml_data['risk']}",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            aml_data["reason"],
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    if aml_data["sentences"]:

        for idx, item in enumerate(
            aml_data["sentences"],
            start=1
        ):

            story.append(
                Paragraph(
                    f"<b>Suspicious Sentence {idx}</b>",
                    styles["Heading3"]
                )
            )

            story.append(
                Paragraph(
                    item["sentence"],
                    styles["BodyText"]
                )
            )

            story.append(
                Paragraph(
                    f"Semantic Similarity Score: "
                    f"{item['score']}",
                    styles["Italic"]
                )
            )

            story.append(
                Spacer(1, 15)
            )

    else:

        story.append(
            Paragraph(
                "No suspicious sentences detected.",
                styles["BodyText"]
            )
        )

    doc.build(story)

    buffer.seek(0)

    return buffer

# ======================================================
# 📜 SAFE HISTORY
# ======================================================

def add_history(entry):

    existing = [
        h["file"]
        for h in st.session_state.history
    ]

    if entry["file"] not in existing:

        st.session_state.history.append(entry)

# ======================================================
# 📂 FILE UPLOAD
# ======================================================

uploaded = st.file_uploader(
    "📂 Upload Financial Policy PDF",
    type=["pdf"]
)

# ======================================================
# 🚀 PROCESS DOCUMENT
# ======================================================

if uploaded:

    pdf_doc = fitz.open(
        stream=uploaded.read(),
        filetype="pdf"
    )

    text = " ".join(
        [p.get_text() for p in pdf_doc[:5]]
    )

    # ==================================================
    # 🚨 VALIDATION
    # ==================================================

    if not validate_document(text):

        st.error(
            "❌ Invalid Financial Document.\n\n"
            "Please upload a valid "
            "FMR / financial policy report."
        )

        st.stop()

    # ==================================================
    # 🔍 RAG SEARCH
    # ==================================================

    results = vectorstore.similarity_search(
        text,
        k=10
    )

    # ==================================================
    # 🧠 AGENTS
    # ==================================================

    policies = fmr_agent(results)

    analysis = analyze(policies)

    aml = aml_agent(text)

    # ==================================================
    # 🎯 LEVEL
    # ==================================================

    if analysis["score"] > 0.8:

        level = "Compliant"

    elif analysis["score"] > 0.5:

        level = "Partially Compliant"

    else:

        level = "Non-Compliant"

    # ==================================================
    # 🧠 DECISION
    # ==================================================

    if aml["risk"] == "Low Risk" and level == "Compliant":

        decision = "APPROVED"

    elif aml["risk"] == "High Risk":

        decision = "REJECTED"

    else:

        decision = "CONDITIONAL"

    # ==================================================
    # 🧾 REPORT
    # ==================================================

    report = agreement(
        analysis,
        aml,
        level,
        decision
    )

    st.session_state.reports[
        uploaded.name
    ] = report

    # ==================================================
    # 📜 HISTORY
    # ==================================================

    add_history({

        "file":
            uploaded.name,

        "score":
            analysis["score"],

        "level":
            level,

        "risk":
            aml["risk"],

        "aml_reason":
            aml["reason"],

        "compliant_docs":
            [
                p["policy"]
                for p in analysis["compliant"]
            ],

        "partial_docs":
            [
                p["policy"]
                for p in analysis["partial"]
            ],

        "non_docs":
            [
                p["policy"]
                for p in analysis["non"]
            ]
    })

    # ==================================================
    # 📊 DASHBOARD
    # ==================================================

    st.subheader("📊 Compliance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Compliance Score",
        round(analysis["score"],2)
    )

    col2.metric(
        "Compliance Level",
        level
    )

    col3.metric(
        "AML Risk",
        aml["risk"]
    )

    chart(analysis)

    # ==================================================
    # 📌 POLICY BREAKDOWN
    # ==================================================

    st.subheader("📌 Policy Breakdown")

    with st.expander("✔ Compliant Policies"):

        if analysis["compliant"]:

            for p in analysis["compliant"]:

                st.success(
                    f"{p['policy']} "
                    f"(Confidence: {p['confidence']})"
                )

        else:

            st.info("No compliant policies")

    with st.expander("⚠ Partial Policies"):

        if analysis["partial"]:

            for p in analysis["partial"]:

                st.warning(
                    f"{p['policy']} "
                    f"(Confidence: {p['confidence']})"
                )

        else:

            st.info("No partial policies")

    with st.expander("❌ Non-Compliant Policies"):

        if analysis["non"]:

            for p in analysis["non"]:

                st.error(
                    f"{p['policy']} "
                    f"(Confidence: {p['confidence']})"
                )

        else:

            st.info("No non-compliant policies")

    # ==================================================
    # 🚨 AML DETECTED SENTENCES
    # ==================================================

    st.subheader("🚨 AML Suspicious Sentences")

    if aml["sentences"]:

        for item in aml["sentences"]:

            st.warning(
                f"{item['sentence']}"
            )

            st.caption(
                f"Semantic Similarity Score: "
                f"{item['score']}"
            )

    else:

        st.success(
            "No suspicious AML content detected."
        )

    # ==================================================
    # 🧾 REPORT VIEWER
    # ==================================================

    st.subheader("🧾 Agreement Report Viewer")

    selected = st.selectbox(
        "Select Report",
        list(st.session_state.reports.keys())
    )

    if selected:

        st.text_area(
            "Generated Report",
            st.session_state.reports[selected],
            height=400
        )

    # ==================================================
    # 📥 MAIN PDF DOWNLOAD
    # ==================================================

    pdf_buffer = generate_pdf(report)

    st.download_button(

        label="📥 Download Final Agreement",

        data=pdf_buffer,

        file_name="Final_Compliance_Report.pdf",

        mime="application/pdf"
    )

    # ==================================================
    # 🚨 AML REPORT DOWNLOAD
    # ==================================================

    aml_pdf = generate_aml_report(aml)

    st.download_button(

        label="🚨 Download AML Highlight Report",

        data=aml_pdf,

        file_name="AML_Highlight_Report.pdf",

        mime="application/pdf"
    )

# ======================================================
# 📜 SIDEBAR HISTORY
# ======================================================

st.sidebar.title("📜 Document Timeline")

if st.session_state.history:

    for h in st.session_state.history:

        with st.sidebar.expander(
            f"📄 {h['file']}"
        ):

            st.write(
                f"📊 Score: "
                f"{round(h['score'],2)}"
            )

            st.write(
                f"🏛 Level: "
                f"{h['level']}"
            )

            st.write(
                f"🚨 AML Risk: "
                f"{h['risk']}"
            )

            st.info(
                h["aml_reason"]
            )

            st.write(
                "✔ Compliant Policies"
            )

            if h["compliant_docs"]:

                for d in h["compliant_docs"]:
                    st.success(d)

            else:
                st.write("None")

            st.write(
                "⚠ Partial Policies"
            )

            if h["partial_docs"]:

                for d in h["partial_docs"]:
                    st.warning(d)

            else:
                st.write("None")

            st.write(
                "❌ Non-Compliant Policies"
            )

            if h["non_docs"]:

                for d in h["non_docs"]:
                    st.error(d)

            else:
                st.write("None")

else:

    st.sidebar.info(
        "No uploaded documents yet."
    )