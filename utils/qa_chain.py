from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # Replaced ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Tuple, List, Dict, Any
import os

def initialize_qa_chain(vectorstore, groq_client=None):
    """Initialize the QA chain with custom prompt."""
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always provide the answer with reference to the specific part of the document it came from.

    {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    llm = ChatGroq(
        model_name= "llama3-70b-8192",  # Fast and capable model
        temperature=0,
        groq_api_key=groq_client.api_key if groq_client else os.getenv("GROQ_API_KEY")
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def generate_challenge_questions(documents: list, groq_client=None, num_questions: int = 3) -> List[Dict[str, str]]:
    """Generate logic-based comprehension questions from the document."""
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    
    prompt_template = """Based on the following document content, generate {num_questions} 
    logic-based or comprehension-focused questions that test deep understanding of the material. 
    For each question, also provide the expected answer with reference to the document.
    
    Document Content:
    {document_content}
    
    Format your response as:
    Question 1: [question text]
    Answer 1: [answer text] (Reference: [specific part of document])
    ---
    Question 2: [question text]
    Answer 2: [answer text] (Reference: [specific part of document])
    ---
    ..."""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["document_content", "num_questions"]
    )
    
    llm = ChatGroq(
        temperature=0.7,
        model_name="llama3-70b-8192",  # Better for complex question generation
        groq_api_key=groq_client.api_key if groq_client else os.getenv("GROQ_API_KEY")
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    # Use first few chunks to generate questions
    document_content = "\n\n".join([doc.page_content for doc in documents[:3]])
    result = llm_chain.run({
        "document_content": document_content,
        "num_questions": num_questions
    })
    
    # Parse the result into questions and answers
    questions = []
    for qa_pair in result.split("---"):
        if "Question" in qa_pair and "Answer" in qa_pair:
            lines = qa_pair.strip().split("\n")
            question = lines[0].split(": ", 1)[1]
            answer = lines[1].split(": ", 1)[1]
            questions.append({"question": question, "answer": answer})
    
    return questions[:num_questions]

def evaluate_user_answer(correct_answer: str, user_answer: str, groq_client=None) -> Tuple[bool, str]:
    """Evaluate user's answer against the correct answer."""
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    
    prompt_template = """Evaluate if the user's answer is correct compared to the expected answer.
    Provide feedback on what they got right or wrong and explain why with reference to the document.
    
    Expected Answer: {correct_answer}
    User's Answer: {user_answer}
    
    Evaluation:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["correct_answer", "user_answer"]
    )
    
    llm = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",  # Good for evaluation
        groq_api_key=groq_client.api_key if groq_client else os.getenv("GROQ_API_KEY")
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    evaluation = llm_chain.run({
        "correct_answer": correct_answer,
        "user_answer": user_answer
    })
    
    # Improved correctness check
    is_correct = any(word in evaluation.lower() for word in [
        "correct", "right", "accurate", "matches", "agrees"
    ]) and not any(word in evaluation.lower() for word in [
        "incorrect", "wrong", "inaccurate", "does not match"
    ])
    return is_correct, evaluation