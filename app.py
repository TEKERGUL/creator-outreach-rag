import os
from dotenv import load_dotenv
from rag_pipeline import CreatorRAG

def main():
    print("Initializing Creator Outreach RAG System...")
    load_dotenv()
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not found in environment. Please set it in .env file.")
        print("Running in mock mode...")
        mock_mode = True
    else:
        mock_mode = False

    rag = CreatorRAG("data/creators_data.csv")
    
    # Simulate User Input (Could be from a Streamlit/Gradio UI)
    print("\n[User Input] Querying vector database for creators...")
    user_query = "Find me makeup creators with high engagement for a new Fenty Beauty campaign."
    print(f"Query: '{user_query}'")
    
    # 1. Retrieval Phase
    matched_creators = rag.search_creators(user_query)
    print(f"\n[Retrieval] Found {len(matched_creators)} matching creators.")
    for c in matched_creators:
        print(f" - {c['username']} (Engagement: {c['engagement_rate']}%)")
        
    # 2. Generation Phase
    if not mock_mode:
        print("\n[Generation] Using Claude to write personalized outreach emails...")
        for creator in matched_creators:
            print(f"\n--- Outreach Email for {creator['username']} ---")
            email = rag.generate_outreach(creator, campaign_goal="Launch our new summer gloss affiliate program")
            print(email)
    else:
        print("\n[Generation] *(Mock output)* Subject: Collaboration Opportunity with Pear Growth...")
        print("*(Add a valid Anthropic API key to see real generated emails)*")

if __name__ == "__main__":
    main()
