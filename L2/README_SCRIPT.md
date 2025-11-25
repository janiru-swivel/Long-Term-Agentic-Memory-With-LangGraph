# Email Assistant - Python Script

This is a working Python script version of the Email Assistant from the Jupyter notebook.

## Features

- **Email Classification**: Automatically categorizes emails into:

  - `IGNORE`: Marketing, spam, mass announcements
  - `NOTIFY`: Important info that doesn't need response
  - `RESPOND`: Emails requiring direct response

- **AI-Powered Response**: Uses LangGraph and OpenAI to draft intelligent email replies

- **Calendar Management**: Schedule meetings and check availability

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in this directory:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the Script

```bash
python email_assistant.py
```

## How It Works

The script demonstrates two examples:

1. **Marketing Email** - Automatically classified as "IGNORE"
2. **Team Question** - Classified as "RESPOND" and generates a reply

The agent uses:

- **Triage Router**: LLM-based classifier that analyzes emails
- **Response Agent**: ReAct agent with tools for email and calendar management
- **LangGraph**: Orchestrates the workflow between components

## Customization

Edit the `profile` and `prompt_instructions` dictionaries in `email_assistant.py` to customize:

- User profile information
- Triage rules
- Agent behavior

## Architecture

```
Email Input → Triage Router → [IGNORE/NOTIFY → END]
                           └→ [RESPOND → Response Agent → Draft Email]
```

The system uses LangGraph's StateGraph to manage the workflow between triaging and responding to emails.
