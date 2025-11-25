# Lesson 3: Email Assistant with Semantic Memory

This is a Python implementation of the email assistant with semantic memory capabilities.

## Features

- **Email Classification**: Automatically triages emails into three categories:

  - IGNORE: Irrelevant emails (newsletters, spam)
  - NOTIFY: Important information without requiring response
  - RESPOND: Emails requiring direct response

- **Semantic Memory**: Remembers details from previous email conversations
- **Smart Actions**:
  - Write and send emails
  - Schedule meetings
  - Check calendar availability

## Setup

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Configure API Keys**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key
   - Add your Anthropic API key

```bash
cp .env.example .env
```

3. **Edit `.env` file** with your actual API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

Run the main script:

```bash
python main.py
```

The script will:

1. Process an initial email about API documentation
2. Process a follow-up email that references the previous conversation
3. Demonstrate memory retrieval capabilities
4. Display stored memories

## Project Structure

```
L3/
├── main.py           # Main script with complete email assistant
├── schemas.py        # Data models and type definitions
├── prompts.py        # System and user prompts
├── utils.py          # Utility functions
├── examples.py       # Example data
├── requirements.txt  # Python dependencies
├── .env.example     # Environment variables template
└── README.md        # This file
```

## How It Works

1. **Triage Router**: Analyzes incoming emails and classifies them
2. **Response Agent**: For emails requiring response, uses ReAct agent with tools
3. **Memory Tools**: Stores and retrieves information from past interactions
4. **LangGraph State Machine**: Manages the workflow between triage and response

## Customization

You can customize the assistant by modifying:

- **User Profile** in `main.py`: Update name, background, etc.
- **Triage Rules** in `main.py`: Adjust classification criteria
- **Agent Instructions**: Modify how the agent uses tools
- **Prompts** in `prompts.py`: Fine-tune system prompts

## Example Output

```
📧 Classification: RESPOND - This email requires a response
--- Agent Response ---
[Agent searches memory, drafts response, optionally schedules meetings]
```

## Notes

- The memory store uses OpenAI embeddings for semantic search
- The response agent uses Claude 3.5 Sonnet for intelligent responses
- Tool calls are placeholders - integrate with real email/calendar APIs for production use
