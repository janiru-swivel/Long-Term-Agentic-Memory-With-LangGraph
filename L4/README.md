# Lesson 4: Email Assistant with Episodic Memory

This is a Python implementation of the email assistant with semantic and episodic memory capabilities.

## Features

- **Email Classification**: Automatically triages emails into three categories
- **Episodic Memory**: Learns from few-shot examples to improve classification
- **Semantic Memory**: Remembers details from previous email conversations
- **Human-in-the-Loop**: Ability to correct misclassifications and improve over time
- **User-Specific Learning**: Each user has their own set of examples

## What's New in Lesson 4

Compared to Lesson 3, this version adds:

- **Few-Shot Learning**: Uses past examples to improve classification accuracy
- **Episodic Memory Store**: Stores examples in a separate "examples" namespace
- **Semantic Search**: Finds similar examples to guide classification
- **Continuous Improvement**: Can add corrected examples to improve future performance

## Setup

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Configure API Keys**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key

```bash
cp .env.example .env
```

3. **Edit `.env` file** with your actual API keys:

```
OPENAI_API_KEY=sk-...
```

## Usage

Run the main script:

```bash
python main.py
```

The script demonstrates:

1. Adding training examples to episodic memory
2. Processing emails without examples (baseline)
3. Adding corrective examples when classification is wrong
4. Re-processing with learned examples (improved accuracy)
5. Testing with similar emails (generalization)
6. User-specific memory (different users have different examples)

## How Episodic Memory Works

1. **Store Examples**: When an email is misclassified, store it with the correct label
2. **Semantic Search**: When processing new emails, search for similar examples
3. **Few-Shot Prompting**: Include similar examples in the prompt
4. **Improved Classification**: LLM follows the examples over general rules

## Project Structure

```
L4/
├── main.py           # Main script with episodic memory
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

- **Adding Examples**: Use `add_example_to_memory()` to add training examples
- **Adjusting Search**: Modify the `limit` parameter in `store.search()`
- **Changing Templates**: Update the few-shot template format
- **User Profiles**: Add different examples for different users

## Example Output

```
✅ Added example to memory with label: ignore
🚫 Classification: IGNORE - This email can be safely ignored

Key Takeaways:
1. Episodic memory allows the assistant to learn from examples
2. Few-shot learning improves classification accuracy
3. Memory is user-specific (harrison vs andrew)
4. Similar emails are handled consistently due to semantic search
```

## Memory Structure

- **Semantic Memory**: `("email_assistant", user_id, "collection")` - General information
- **Episodic Memory**: `("email_assistant", user_id, "examples")` - Few-shot examples

## Notes

- Uses OpenAI GPT-4o for response generation
- Uses OpenAI text-embedding-3-small for semantic search
- Examples are stored with full email content and classification label
- Semantic search ensures similar emails benefit from past examples
