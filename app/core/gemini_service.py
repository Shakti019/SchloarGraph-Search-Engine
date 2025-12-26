import google.generativeai as genai
from app.core.config import settings

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)

def get_gemini_model():
    # Using the stable flash alias which usually has better free tier quotas
    return genai.GenerativeModel('gemini-flash-latest')

async def analyze_paper_content(title: str, abstract: str):
    """
    Generates a detailed analysis of the paper using Gemini.
    """
    model = get_gemini_model()
    prompt = f"""
    You are an expert academic researcher. Please provide a detailed analysis of the following research paper.
    
    Title: {title}
    Abstract: {abstract}
    
    Please structure your response with the following sections using Markdown:
    1. **Core Contribution**: What is the main novelty?
    2. **Key Methodology**: How did they do it?
    3. **Implications**: Why does this matter?
    4. **Potential Limitations**: What might be missing?
    5. **Future Directions**: Where can this go next?
    
    Keep the tone professional and academic.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

async def chat_with_paper_context(history: list, message: str, context: str):
    """
    Handles chat conversation about the paper.
    """
    model = get_gemini_model()
    
    # Construct prompt with context
    system_prompt = f"""
    You are a helpful research assistant discussing a specific paper.
    Context (Paper Abstract): {context}
    
    Answer the user's questions based on this context and your general knowledge.
    If the answer isn't in the abstract, use your general knowledge but mention that it's not explicitly in the provided text.
    """
    
    chat = model.start_chat(history=history)
    
    # We might need to inject context if it's a fresh chat, but for simplicity 
    # we'll just prepend context to the message or rely on the user knowing what they are talking about.
    # Better: Send context in the first message or system instruction if supported.
    # Gemini Pro API supports history.
    
    full_prompt = f"{system_prompt}\n\nUser: {message}"
    
    try:
        response = chat.send_message(full_prompt)
        return response.text
    except Exception as e:
        return f"Error in chat: {str(e)}"

async def expand_query(query: str) -> str:
    """
    Expands the search query with synonyms and related terms using Gemini.
    """
    model = get_gemini_model()
    prompt = f"You are a search optimization assistant. Refine and expand the following search query to improve retrieval of academic papers. Return ONLY the optimized query string, no explanations. Original Query: {query}"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Query expansion failed: {e}")
        return query

