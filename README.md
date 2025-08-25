# 🎶 Dream Composer — AI-Powered Dream-to-Music Generator  

## 📖 Project Overview  

**Dream Composer** is an AI-powered application that transforms users’ dreams into personalized music.  
By analyzing dream descriptions through natural language processing (NLP) and mapping emotions to music theory, the system generates melodies, sheet notation, and mood-based visualizations.  

This project demonstrates programming principles of **Correctness, Efficiency, and Scalability** while keeping the scope practical and achievable.  

---

## 🚀 Core Objectives  

### ✅ Correctness  
- Map dream descriptions to emotions and music parameters (tempo, key, instruments).  
- Generate music and visualizations that consistently reflect the interpreted mood.  

### ⚡ Efficiency  
- Optimized API requests for NLP and music generation.  
- Use caching strategies for repeated requests to reduce re-processing.  

### 📈 Scalability  
- Current implementation: single-service API.  
- Roadmap: modular design (analysis, generation, visualization) that can evolve into microservices for higher traffic.  

---

## 🛠️ Tech Stack  

### Current Choices  
- **Frontend:** React.js (dream input, playback, visualization UI)  
- **Backend:** Node.js + Express (API handling, NLP integration)  
- **NLP & AI:** Hugging Face / OpenAI APIs for emotion & meaning extraction  
- **Music Generation:** Magenta.js (for melody/MIDI generation)  
- **Database:** MongoDB (storing user dreams & generated outputs)  

### Roadmap (Optional Enhancements)  
- Music21 or ABC notation for advanced sheet music export  
- Cloud deployment (AWS/GCP + Docker) for scaling  
- Retrieval-Augmented Generation (RAG) with psychology & music theory resources  

---

## ⚙️ Implementation Plan  

1. **Input Handling**  
   - User enters or records dream description.  
   - NLP extracts mood, symbols, and emotional context.  

2. **Dream Analysis**  
   - Map moods → musical elements (tempo, scale, instrumentation).  

3. **Music Generation**  
   - Magenta.js composes melodies based on mapped parameters.  
   - Export to MIDI and provide playback.  

4. **Visualization**  
   - Emotions mapped to color gradients + simple waveform animations.  

5. **Output Delivery**  
   - 🎧 Play dream-inspired audio  
   - 📝 View/download sheet music (future roadmap)  
   - 🌈 Watch mood-based visualization  

---

## 📊 Example Workflow  

**Input Dream:**  
> “I dreamt I was flying over a shimmering city, feeling happy and nostalgic.”  

**AI Processing:**  
- Mood: Joy + Nostalgia  
- Tempo: 95 BPM (calm, flowing)  
- Instruments: Piano + Strings  
- Visualization: Blue-gold gradient  

**Output:**  
- 🎶 Calm piano/strings melody (MIDI playback)  
- 🌈 Animated visualization  

---

## 🔑 Evaluation Alignment  

- **Correctness:** Moods are mapped consistently to defined music rules.  
- **Efficiency:** API requests and results cached for faster response.  
- **Scalability:** Current single-service backend can expand into modular components.  

---

## 📜 License  

- **Project License:** MIT (ensure `LICENSE` file is included in repo).  
- **Third-Party Tools & APIs:**  
  - OpenAI/Hugging Face APIs → subject to their usage terms.  
  - Magenta.js (Apache 2.0).  
  - Future integrations (e.g., Music21) may have additional requirements.  

Contributors must review and comply with respective licenses.  

---

## 🎥 Video Explanation (For Submission)  

In the demo video:  
1. Introduce the idea (dream → music).  
2. Show how NLP maps dream emotions to music elements.  
3. Explain Correctness, Efficiency, and Scalability principles.  
4. Demo one input/output example.  

---

✨ **Dream big, compose bigger!** ✨  
