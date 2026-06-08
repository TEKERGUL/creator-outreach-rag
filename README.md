# Creator Outreach RAG System 🎯✉️

An internal tool designed to automate influencer discovery and personalized outreach using **Retrieval-Augmented Generation (RAG)** and the **Claude API**.

Instead of manually searching through spreadsheets and copying/pasting generic email templates, this tool allows marketing managers to query the creator database using natural language and automatically generate highly personalized outreach emails.

## 🚀 Why this exists (Agency Pain Point)
Finding the right creators and crafting personalized emails that actually get responses is the biggest bottleneck in influencer marketing. This system acts as an "Agentic CRM assistant," scanning past performance data to find perfect matches and drafting emails tailored to the creator's specific niche and past brand deals.

## 🛠 Tech Stack
- **Python (Pandas)**: For managing the creator database (simulating a Vector DB).
- **Anthropic Claude 3 API**: For context-aware email generation.
- **RAG Architecture**: Retrieves relevant creator context before asking the LLM to generate content.

## 💻 How it works
1. **Retrieval**: User enters a query like *"Find makeup creators with >10% engagement rate."* The system searches `creators_data.csv`.
2. **Augmentation**: The matched creator profiles are injected into the context window for Claude.
3. **Generation**: Claude uses the profile data (past sponsors, content style) to write a personalized outreach email asking them to join the affiliate program.

## ⚙️ Setup Instructions
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the root directory and add your Anthropic API Key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```
4. Run the application: `python app.py`

## 📈 Future Roadmap
- Replace Pandas search with a true Vector Database (ChromaDB / Pinecone).
- Connect directly to the HubSpot API to automatically log sent emails and track creator responses in the CRM pipeline.
