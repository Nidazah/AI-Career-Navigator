#  AI Career Navigator

An AI-powered career guidance web app that analyzes your CV, identifies skill gaps, recommends career paths, and provides a personalized learning roadmap — all for free using Groq's LLaMA model.

---

##  Features

-  **CV Analysis** — Upload your resume (PDF or TXT) and get instant AI analysis
-  **Skill Detection** — Identifies your current skills from your CV
-  **Skill Gap Analysis** — Shows what skills you're missing for your target roles
-  **Career Path Recommendations** — Suggests roles that match your profile
-  **Learning Roadmap** — Step-by-step plan with time estimates to reach your goal
-  **Skill Score Chart** — Visual bar chart comparing current vs target proficiency
-  **AI Career Chatbot** — Ask anything about interviews, resources, or career advice

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, Tailwind CSS, Chart.js |
| Backend | Python, FastAPI, Uvicorn |
| AI Model | LLaMA 3.1 (via Groq API — Free) |
| PDF Parsing | pdfplumber |

---

##  Project Structure

```
career-navigator/
├── index.html          # Frontend UI
├── .env.example        # Environment variable template
├── .gitignore
└── backend/
    ├── main.py         # FastAPI backend
    └── requirements.txt
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Nidazah/AI-Career-Navigator.git
cd AI-Career-Navigator
```

### 2. Get a Free Groq API Key
- Go to [https://console.groq.com](https://console.groq.com)
- Sign up (no credit card needed)
- Create an API key

### 3. Set up environment variables
```bash
# Copy the example file
cp .env.example .env

# Open .env and paste your Groq API key
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 5. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 6. Start the backend server
```bash
uvicorn main:app --reload
```

### 7. Open the app
Double-click `index.html` to open it in your browser.

>  Keep the terminal with uvicorn running while using the app.

---

##  Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your free API key from [console.groq.com](https://console.groq.com) |

---

##  Limitations

- CV must be in **PDF or TXT** format (max 2MB)
- Requires **internet connection** to call Groq API
- Server must be running locally before using the app

---

##  Future Improvements

- [ ] Add support for DOCX files
- [ ] Save analysis history
- [ ] Export roadmap as PDF
- [ ] Deploy backend to cloud (Render/Railway)
- [ ] Add user authentication

---

##  Author

**Nida** — CS Student | AI Enthusiast | Freelancer  
 [GitHub](https://github.com/Nidazah) • [LinkedIn](https://linkedin.com/in/nidazahra24)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
