import os
import pandas as pd
from anthropic import Anthropic

class CreatorRAG:
    def __init__(self, data_path: str):
        # In a real app, this would use ChromaDB or Pinecone.
        # For this demo, we simulate a vector search with Pandas to keep it lightweight.
        self.df = pd.read_csv(data_path)
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-3-haiku-20240307"

    def search_creators(self, query: str):
        """
        Simulates retrieving relevant context from a Vector DB based on user query.
        """
        # A simple keyword match to simulate RAG retrieval
        query_lower = query.lower()
        if "makeup" in query_lower or "beauty" in query_lower:
            results = self.df[self.df['niche'] == 'makeup']
        elif "skincare" in query_lower:
            results = self.df[self.df['niche'] == 'skincare']
        else:
            results = self.df.head(3) # fallback
            
        return results.to_dict('records')

    def generate_outreach(self, creator: dict, campaign_goal: str) -> str:
        """
        Uses Claude API to generate a personalized outreach email based on the creator's profile.
        """
        prompt = f"""
        You are an Influencer Marketing Manager at Pear Growth.
        Write a personalized outreach email to the following creator:
        
        Creator: {creator['username']}
        Niche: {creator['niche']}
        Engagement Rate: {creator['engagement_rate']}%
        Past Sponsors: {creator['past_sponsors']}
        Content Style: {creator['content_style']}
        
        Campaign Goal: {campaign_goal}
        
        Keep the email short, compelling, and professional. We want to invite them to our affiliate program.
        """
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
