# app.py
import streamlit as st
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Groq-compatible OpenAI client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📄 Smart Research Assistant")

file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])

if file:
    def extract_text(file):
        if file.name.endswith(".pdf"):
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in pdf])
        else:
            return file.read().decode("utf-8")

    document_text = extract_text(file)

    # --- Auto Summary ---
    with st.spinner("Generating summary..."):
        try:
            summary_resp = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "Summarize this document in under 150 words."},
                    {"role": "user", "content": document_text[:3000]}
                ]
            )
            summary = summary_resp.choices[0].message.content
            st.success("Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Summary generation failed: {e}")
            st.stop()

    mode = st.radio("Choose a mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        question = st.text_input("Ask a question based on the document")
        if question:
            with st.spinner("Answering..."):
                try:
                    answer = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "Answer based only on the content of the document. Include justification (e.g., 'as stated in section 2')"},
                            {"role": "user", "content": f"Document:\n{document_text[:4000]}\n\nQuestion: {question}"}
                        ]
                    )
                    st.success("Answer")
                    st.write(answer.choices[0].message.content)
                except Exception as e:
                    st.error(f"Failed to get answer: {e}")

    elif mode == "Challenge Me":
        with st.spinner("Generating questions..."):
            try:
                q_resp = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "Generate 3 logic-based comprehension questions based on the following document."},
                        {"role": "user", "content": document_text[:3000]}
                    ]
                )
                raw_questions = q_resp.choices[0].message.content.strip()
                questions = raw_questions.split("\n")
                questions = [q for q in questions if q.strip() != ""]
            except Exception as e:
                st.error(f"Question generation failed: {e}")
                st.stop()

        answers = []
        for idx, q in enumerate(questions):
            st.markdown(f"**{q}**")
            ans = st.text_input(f"Your Answer {idx+1}", key=f"ans_{idx}")
            answers.append((q, ans))

        if st.button("Evaluate My Answers"):
            with st.spinner("Evaluating..."):
                evaluation_prompt = ""
                for q, a in answers:
                    evaluation_prompt += f"Question: {q}\nUser Answer: {a}\n\n"

                try:
                    eval_resp = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "Evaluate the answers strictly based on document content. Provide feedback and justification with paragraph or section reference."},
                            {"role": "user", "content": f"Document:\n{document_text[:4000]}\n\n{evaluation_prompt}"}
                        ]
                    )
                    feedback = eval_resp.choices[0].message.content
                    st.success("Feedback")
                    st.write(feedback)
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
