import json
import asyncio
from dotenv import load_dotenv
import streamlit as st

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from arxiv_helper import get_arxiv_papers

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()


# ---------------- MODELS ---------------- #

class Paper(BaseModel):
    title: str
    authors: list[str]
    year: int
    abstract: str
    publication_date: str
    journal: str


class Formula(BaseModel):
    name: str
    latex: str
    description: str
    formula: str


class Trend(BaseModel):
    topic: str
    description: str
    examples: list[str]


class Report(BaseModel):
    topic: str
    papers: list[Paper]
    formulas: list[Formula]
    trends: list[Trend]


# ---------------- PDF ---------------- #

def create_pdf(papers):
    doc = SimpleDocTemplate("research_report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("AI Research Report", styles["Title"]))
    content.append(Spacer(1, 12))

    for p in papers:
        content.append(Paragraph(p["title"], styles["Heading2"]))
        content.append(Paragraph(f"Authors: {', '.join(p['authors'])}", styles["Normal"]))
        content.append(Paragraph(f"Published: {p['published']}", styles["Normal"]))
        content.append(Paragraph(p["summary"], styles["BodyText"]))
        content.append(Spacer(1, 12))

    doc.build(content)


# ---------------- GEMINI ---------------- #

async def generate_report(topic, research_questions, timeframe):

    task = f"""
Topic: {topic}
Research Questions: {research_questions}
Timeframe: {timeframe}

Generate structured research insights with papers, formulas, and trends.
"""

    model = (
        init_chat_model(
            "gemini-3.1-flash-lite",
            model_provider="google-genai",
        )
        .with_structured_output(Report)
    )

    result = await model.ainvoke([
        {
            "role": "system",
            "content": "You are a research assistant that generates structured academic insights."
        },
        {"role": "user", "content": task},
    ])

    return result


# ---------------- UI ---------------- #

st.set_page_config(
    page_title="AI Research Assistant Pro",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Assistant Pro")
st.write("Real arXiv papers + AI-powered research analysis")

topic = st.text_input("Research Topic")
research_questions = st.text_area("Research Questions", height=120)
timeframe = st.text_input("Research Timeframe", placeholder="2020-2026")


# ---------------- MAIN ---------------- #

if st.button("Generate Research Report"):

    if topic and research_questions and timeframe:

        # 🔥 STEP 1: FETCH PAPERS
        with st.spinner("Fetching real research papers from arXiv..."):
            papers = get_arxiv_papers(topic, 5)

        st.success("Real papers fetched successfully!")

        st.subheader("📚 Research Papers")

        for p in papers:
            st.markdown(f"### {p['title']}")
            st.write("👨‍🔬 Authors:", ", ".join(p["authors"]))
            st.write("📅 Published:", p["published"])
            st.info(p["summary"])
            st.markdown(f"🔗 [Open PDF]({p['pdf_url']})")
            st.markdown("---")

        # 🔥 STEP 2: AI INSIGHTS
        st.subheader("🤖 AI Research Insights")

        with st.spinner("Generating AI insights..."):
            report = asyncio.run(
                generate_report(topic, research_questions, timeframe)
            )

        st.write("**Topic:**", report.topic)

        # Papers
        st.subheader("🧠 Structured Insights")

        for paper in report.papers:
            with st.expander(paper.title):
                st.write("Authors:", ", ".join(paper.authors))
                st.write("Journal:", paper.journal)
                st.write("Date:", paper.publication_date)
                st.write(paper.abstract)

        # Formulas
        st.subheader("📐 Important Formulas")

        for formula in report.formulas:
            st.markdown(f"### {formula.name}")
            st.code(formula.formula)
            st.write(formula.description)

        # Trends
        st.subheader("📊 Research Trends")

        for trend in report.trends:
            st.markdown(f"### {trend.topic}")
            st.write(trend.description)
            for ex in trend.examples:
                st.write("•", ex)

        # ---------------- DOWNLOADS ---------------- #

        create_pdf(papers)

        with open("research_report.pdf", "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name="AI_Research_Report.pdf"
            )

        json_data = json.dumps(report.model_dump(), indent=4)

        st.download_button(
            "📥 Download JSON Report",
            data=json_data,
            file_name="research_report.json",
            mime="application/json"
        )

    else:
        st.warning("Please fill all fields.")