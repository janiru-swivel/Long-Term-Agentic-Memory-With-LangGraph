# Lesson 5: Email Assistant with Procedural Memory

This is a Python implementation of the email assistant with semantic, episodic, and procedural memory capabilities.

## Features

- **Email Classification**: Automatically triages emails into three categories
- **Episodic Memory**: Learns from few-shot examples to improve classification
- **Semantic Memory**: Remembers details from previous email conversations
- **Procedural Memory**: Dynamically updates its own instructions based on feedback
- **Self-Improving Agent**: Automatically adapts behavior based on user preferences

## What's New in Lesson 5

Compared to Lesson 4, this version adds:

- **Procedural Memory**: Stores and updates agent instructions dynamically
- **Multi-Prompt Optimizer**: Uses LLM to update prompts based on feedback
- **Self-Modification**: Agent can improve its own behavior over time
- **User Preferences**: Learns user-specific preferences for email handling and writing style

## Memory Types

1. **Semantic Memory** (`"collection"`): Remembers facts and details from conversations
2. **Episodic Memory** (`"examples"`): Stores few-shot examples for classification
3. **Procedural Memory** (user namespace): Stores instructions and rules
   - `agent_instructions`: How to use tools (email writing, scheduling)
   - `triage_ignore`: Rules for ignoring emails
   - `triage_notify`: Rules for notification-only emails
   - `triage_respond`: Rules for emails requiring response

## Setup

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Configure API Keys**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key
   - Add your Anthropic API key (required for prompt optimization)

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

The script demonstrates:

1. Processing emails with default instructions
2. Providing feedback: "Always sign your emails `John Doe`"
3. Re-processing with updated agent instructions
4. Providing feedback: "Ignore any emails from Alice Jones"
5. Re-processing with updated triage rules

## How Procedural Memory Works

1. **Initial Processing**: Agent uses default instructions from code
2. **Feedback Loop**: User provides feedback on agent behavior
3. **Prompt Optimization**: LLM analyzes feedback and updates relevant prompts
4. **Automatic Application**: Updated prompts are stored and used immediately
5. **Continuous Improvement**: Agent behavior evolves based on accumulated feedback

## Example Workflow

```python
# 1. Process email (default behavior)
response = email_agent.invoke({"email_input": email}, config)

# 2. Provide feedback
feedback = "Always sign your emails `John Doe`"

# 3. Update procedural memory
update_prompts_with_feedback(store, "lance", response['messages'], feedback)

# 4. Re-process (new behavior applied automatically)
response = email_agent.invoke({"email_input": email}, config)
```

## Project Structure

```
L5/
├── main.py           # Main script with procedural memory
├── schemas.py        # Data models and type definitions
├── prompts.py        # System and user prompts
├── utils.py          # Utility functions
├── examples.py       # Example data
├── requirements.txt  # Python dependencies
├── .env.example     # Environment variables template
└── README.md        # This file
```

## Customization

You can customize the assistant by:

- **Adding Feedback**: Use `update_prompts_with_feedback()` with any natural language feedback
- **Viewing Memory**: Use `display_memory_state()` to see current instructions
- **Manual Updates**: Directly modify prompts in the store using `store.put()`
- **Optimizer Settings**: Adjust the `update_instructions` and `when_to_update` fields

## Example Output

```
✅ Updated main_agent
   Old: Use these tools when appropriate to help manage John's tasks efficiently...
   New: Use these tools when appropriate to help manage John's tasks efficiently. Always sign emails with `John Doe`...

✅ Updated triage-ignore
   Old: Marketing newsletters, spam emails, mass company announcements...
   New: Marketing newsletters, spam emails, mass company announcements, emails from Alice Jones...

Key Takeaways:
1. Procedural memory allows the agent to update its own instructions
2. Feedback is automatically translated into prompt updates
3. Both agent behavior and triage rules can be dynamically modified
4. The agent learns and adapts based on user preferences
```

## Memory Storage

Procedural memory is stored in user-specific namespaces:

- `(user_id,) + "agent_instructions"`: Agent tool usage instructions
- `(user_id,) + "triage_ignore"`: Email ignore rules
- `(user_id,) + "triage_notify"`: Email notification rules
- `(user_id,) + "triage_respond"`: Email response rules

## Notes

- Requires both OpenAI (for agent) and Anthropic (for prompt optimization) API keys
- Prompt optimizer uses Claude 3.5 Sonnet for intelligent instruction updates
- Instructions are user-specific and persistent across sessions
- The optimizer automatically determines which prompts need updating based on feedback
