# VoiceQuest AI

VoiceQuest AI is a conversational AI system that integrates voice recognition and Retrieval-Augmented Generation (RAG) methods to provide context-based responses to users' voice queries. The system allows users to ask questions using their voice, and the AI responds with accurate and contextually relevant information.

## Features:
- **Voice-powered interaction**: Use your voice to interact with the system.
- **Retrieval-Augmented Generation (RAG)**: Answers are generated based on retrieved relevant documents or information.
- **Text-to-Speech (TTS) and Speech-to-Text (STT)**: Uses **Faster-Whisper** for speech processing to convert user voice to text and **gTTs** to convert the generated text to speech.
- **Context-aware responses**: The model provides detailed answers relevant to the context of the user's query.

## Setup Instructions:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Keerthanadevaraj11/VoiceQuest-AI.git
   ```

2. **Create a virtual environment and activate it**:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

**3. Install dependencies**:
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**:
Create a .env file in the root directory of the project and add the following environment variables with your own API keys:

```bash
LLAMA_CLOUD_API_KEY=<your-llama-cloud-api-key>
GOOGLE_API_KEY=<your-google-api-key>
GROQ_API_KEY=<your-groq-api-key>
```

**5. Run the application**:
```bash
python app.py
```
