# Stratagem AI: Game Theory Strategy Simulator

**Live Demo:** [stratagem.streamlit.app](https://stratagem.streamlit.app)

## 📌 Executive Summary
Stratagem AI is a competitive strategy tool designed to bridge the gap between mathematical game theory and executive decision-making. By combining **Nashpy** for quantitative equilibrium modeling and **Gemini 1.5 Pro** for qualitative risk assessment, the app allows users to simulate duopolistic competition (e.g., pricing wars, market entry) and receive a strategic briefing from an "AI Rival CEO."

---

## 🚀 Key Features
* **Nash Equilibrium Engine:** Solves 2x2 payoff matrices for Pure and Mixed strategies using the Lemke-Howson algorithm via `nashpy`.
* **AI Strategic Analysis:** Integrated Google Gemini 1.5 API to translate mathematical outcomes into "Consultant-speak" briefings on market elasticity and predatory risks.
* **Dynamic Visualizations:** Interactive Plotly heatmaps that highlight profit territories and equilibrium stability.
* **Bloomberg-Inspired UI:** A high-contrast, professional dark-themed dashboard built with Streamlit for C-suite usability.

---

## 🛠️ Technical Stack
* **Language:** Python 3.10+
* **Framework:** Streamlit (Web UI & Deployment)
* **Game Theory:** Nashpy (Matrix Games)
* **AI/LLM:** Google Gemini 1.5 Pro (Strategic Persona)
* **Visualization:** Plotly (Heatmaps & Metric Cards)

---

## 📖 How to Use
1.  **Input Payoffs:** Enter the numerical benefits for both the 'User' and the 'Rival' in the 2x2 matrix.
2.  **API Integration:** Provide a Gemini API Key in the sidebar to unlock the Rival CEO analysis.
3.  **Solve:** Click the **"Solve Nash Equilibrium"** button to identify stable strategies.
4.  **Analyze:** Review the expected payoffs and the AI's qualitative assessment of the competitive landscape.

---

## 🎓 About the Developer
**Anish G.** *Computer Science & Economics | Vanderbilt University* Focused on the intersection of Algorithmic Game Theory and Management Consulting.
