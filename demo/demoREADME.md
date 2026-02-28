# 🎭 AI Emotion Analysis Demo

A lightweight, ethical demo of facial emotion recognition using Groq API + Llama 4 Scout.

## ⚠️ Ethical Notice
This demo is for educational purposes only. AI emotion recognition has documented biases and should not be used for high-stakes decisions. See our research: [AI-Based-Emotion-Detection](https://github.com/Jzaman2004/AI-Based-Emotion-Detection)

## 🚀 Quick Start

1. **Clone & Navigate**
   ```bash
   cd demo
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Run Locally**
   ```bash
   python app.py
   ```
   - Codespaces: Click "Open in Browser" when prompted (port 5000)
   - Local: Visit `http://localhost:5000`

4. **Use the Demo**
   - Allow camera access on the capture page
   - Position face in frame
   - Click "Capture & Analyze"
   - You will be redirected to a results page
   - On results, use **Share Your Perspective** to vote on ethical questions
   - Each question has separate **Yes** and **No** vote buttons
   - Each vote increments the matching **Yes** or **No** counter live
   - Counters persist only for the current browser session (`sessionStorage`)
   - Click "Retry" on results page to return to capture page

## 💬 Ethical Feedback Feature
- Located on the results page, below the **Retry** button
- Includes multiple ethical perspective questions, each with **Yes**/**No** vote buttons
- Each click increases the corresponding **Yes** or **No** counter immediately on the page
- Counter state is session-scoped and resets when session storage is cleared or session ends

## 🔒 Security
- API key is stored in `.env` (never committed)
- Backend proxies all API calls (key never exposed to frontend)
- Images are processed in-memory, not stored

## 📱 Responsive Design
- Capture page: single live camera view
- Results page: captured image + analysis details
- Mobile: touch-friendly single-column layout
