# 🎶 Dream Composer — AI-Powered Dream-to-Music Generator  

## 📖 Project Overview  

**Dream Composer** is an AI-powered application that transforms users’ dreams into personalized music. By analyzing dream descriptions through natural language processing (NLP) and combining insights from psychology and music theory, the system generates a unique piece of music with mood visualizations and sheet notation.  

This project not only focuses on creativity but also demonstrates strong programming principles: **Correctness, Efficiency, and Scalability**.  

---

## 🚀 Core Objectives  

### ✅ Correctness  
- The system accurately interprets dream descriptions.  
- Maps emotions and dream symbols to relevant musical elements (tempo, scale, instruments).  
- Returns a correct music output (MIDI/audio + sheet music + mood visualization) based on the user’s input.  

### ⚡ Efficiency  
- Optimized APIs for dream analysis and music generation.  
- Caching for repeated requests to avoid re-computation.  
- Lightweight audio generation to minimize latency.  

### 📈 Scalability  
- API-first architecture to handle high user traffic.  
- Cloud-based deployment for large-scale music generation.  
- Modular microservices (dream analysis, music generation, visualization) to scale independently.  

---

## 🛠️ Tech Stack  

- **Frontend:** React.js (UI for dream input, playback, and visualization)  
- **Backend:** Node.js + Express (API handling, NLP integration, request routing)  
- **NLP & AI:**  
  - Hugging Face / OpenAI APIs for dream sentiment & meaning extraction  
  - Custom RAG (Retrieval-Augmented Generation) with psychology & music theory references  
- **Music Generation:**  
  - Magenta.js / MIDI.js for composing melodies  
  - Music21 / ABC notation for sheet music export  
- **Database:** MongoDB (to store user dreams, generated music metadata)  
- **Deployment:** Docker + AWS/GCP for scalability  

---

## ⚙️ Implementation Plan  

1. **Input Handling**  
   - Users type or record their dream.  
   - NLP processes the text → extracts mood, symbols, emotions.  

2. **Dream Analysis Engine**  
   - Uses psychology + music theory dataset (RAG) to interpret meaning.  
   - Maps dream emotions → tempo, key, and instrument selection.  

3. **Music Generation Module**  
   - Converts dream analysis → structured music plan (intro, chorus, bridge).  
   - Generates MIDI + renders audio.  
   - Exports sheet music notation.  

4. **Visualization Engine**  
   - Assigns dream emotions → color gradients and waveform animations.  

5. **Output Delivery**  
   - Users can:  
     - 🎧 Listen to audio  
     - 📝 Download sheet music  
     - 🌈 Watch dream mood visualizer  

---

## 📊 Example Workflow  

**Input Dream:**  
> “I dreamt I was flying over a shimmering city, feeling happy and nostalgic.”  

**AI Processing:**  
- Mood: Joy + Nostalgia  
- Tempo: 95 BPM (calm, flowing)  
- Instruments: Piano + Strings + Synth Pads  
- Visualization: Blue-gold gradient with flowing motion  

**Output:**  
- 🎶 Calming piano melody with string harmonies  
- 📝 Sheet music (MIDI export)  
- 🌈 Animated visualization  

---

## 🔑 Evaluation Alignment  

- **Correctness:**  
   - Each dream maps consistently to a defined music plan using structured rules + AI models.  
- **Efficiency:**  
   - Caching results for repeated mood mappings, optimizing audio rendering pipeline.  
- **Scalability:**  
   - API-first modular architecture → backend services (analysis, generation, visualization) can scale separately.  

---

## 📜 License  

MIT License — free for personal and academic use.  

---

## 🎥 Video Explanation (For Submission)  

In the demo video:  
1. Briefly introduce the problem and vision.  
2. Explain the system workflow (Input → Analysis → Music → Output).  
3. Highlight how correctness, efficiency, and scalability are achieved.  
4. Show a short demo example of input/output.  

---

✨ **Dream big, compose bigger!** ✨  
