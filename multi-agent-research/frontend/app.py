import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/research"


st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


st.title("🔬 Multi-Agent AI Research Assistant")

st.write(
    "AI Research Assistant powered by Multi-Agent AI, "
    "RAG, FAISS and Ollama."
)


st.markdown("### Enter your research question")


question = st.text_area(
    "Research Question",
    placeholder=(
        "Example: What are the applications of "
        "artificial intelligence in healthcare?"
    ),
    height=120
)


if st.button("🚀 Generate Research Report"):

    # Remove unnecessary spaces
    cleaned_question = question.strip()

    # Check empty input
    if not cleaned_question:

        st.warning(
            "⚠️ Please enter a research question."
        )

    # Check very short / vague input
    elif len(cleaned_question.split()) < 3:

        st.warning(
            "⚠️ Please enter a more specific research question."
        )

        st.info(
            "Example: What are the applications of "
            "AI in healthcare?"
        )

    else:

        st.info(
            "Research started. Please wait while the "
            "AI agents process your question..."
        )

        with st.spinner(
            "🔄 Researching... This may take several minutes."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": cleaned_question
                    },
                    timeout=None
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "✅ Research completed successfully!"
                    )

                    # Research Plan
                    st.markdown("## 📋 Research Plan")

                    st.write(
                        result.get(
                            "research_plan",
                            "No research plan available."
                        )
                    )

                    # Analysis
                    st.markdown("## 🔍 Analysis")

                    st.write(
                        result.get(
                            "analysis",
                            "No analysis available."
                        )
                    )

                    # Fact Check
                    st.markdown("## ✅ Fact Check")

                    st.write(
                        result.get(
                            "fact_check",
                            "No fact-check information available."
                        )
                    )

                    # Final Report
                    st.markdown(
                        "## 📄 Final Research Report"
                    )

                    st.write(
                        result.get(
                            "final_report",
                            "No final report available."
                        )
                    )

                else:

                    st.error(
                        f"❌ FastAPI returned error "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the FastAPI server."
                )

                st.info(
                    "Make sure FastAPI is running at "
                    "http://127.0.0.1:8000"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "❌ The research request timed out."
                )

                st.info(
                    "The FastAPI backend may still be "
                    "processing the research request."
                )

            except Exception as e:

                st.error(
                    "❌ Unexpected error occurred."
                )

                st.code(
                    str(e)
                )