# Vox AI – AI-Powered Crypto Trading Assistant

Vox AI is an advanced crypto trading assistant that generates actionable signals by combining **technical analysis**, **pattern detection**, and **multi-strategy fusion**. It also uses **Stable Diffusion v1** to generate visual images when predicting the price of a cryptocurrency, giving users a unique and interactive forecasting experience.

---

## 🚀 Features

- **5 Best Trading Strategies**
  - **RSI-EMA Retest** – Confirms trend reversals using RSI and EMA alignment.
  - **Breakout Strategy** – Detects strong price breakouts beyond support/resistance zones.
  - **Fibonacci Retracements** – Identifies pullback levels for precise re-entry.
  - **Fair Value Gaps (FVG)** – Finds price imbalances for potential fills.
  - **Strategy Fusion** – `combine_strategies.py` merges technical and pattern signals into final decisions (*Strong Buy*, *Neutral*, *Strong Sell*).

- **Technical Analysis with TA-Lib 0.6.4**  
  Optimized for high-performance indicators (RSI, EMA, MACD, Bollinger Bands, etc.).

- **Pattern Recognition**  
  `pattern_analyzer.py` detects candlestick and chart formations for deeper insights.

- **Stable Diffusion v1 for Price Prediction Visuals**  
  When users ask Vox AI to predict the price of a cryptocurrency, it generates **AI-driven price forecast images** using Stable Diffusion v1.

- **Real-Time WebSocket Server**  
  Hosted via **`VoxAI/src/interface/server.py`** to stream signals to clients.

- **API Key Management**  
  - **Development:** Keys are stored in `config/credentials.yaml`.  
  - **Production:** Keys must be stored as environment variables.

---

## 🛠 Tech Stack

- **Core:** `numpy 2.2.3`, `pandas 2.2.3`, `ta-lib 0.6.4`
- **AI Image Generation:** Stable Diffusion v1
- **Backend:** Python 3.11+
- **Communication:** WebSocket server for real-time signals
- **Data Sources:** CoinGecko, CoinMarketCap, and other crypto APIs.
-**LLM model:** Llama 4 marvick for nlp response and queries.


## ⚙️ Setup

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/vox-ai.git
cd vox-ai

2. Install dependencies

pip install -r requirements.txt

3. Configure API keys

Development: Place keys in config/credentials.yaml.

Production: Use environment variables for secure deployments.


4. Run the WebSocket server

python src/interface/server.py


---

🔒 Security Note

Never commit config/credentials.yaml to Git.

Use .gitignore to exclude sensitive files.

Use platform-level environment variables (Render, Railway, etc.) in production.



---

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

