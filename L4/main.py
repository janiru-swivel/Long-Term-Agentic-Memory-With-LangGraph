"""
Lesson 4: Email Assistant with Semantic + Episodic Memory

This script implements an email assistant that:
- Classifies incoming messages (respond, ignore, notify)
- Drafts responses
- Schedules meetings
- Uses semantic memory to remember details from previous emails
- Uses episodic memory (few-shot examples) to improve classification accuracy
"""

import os
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.types import Command

from prompts import triage_system_prompt, triage_user_prompt


# Load environment variables
load_dotenv()


# User profile configuration
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# Prompt instructions
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}


# Define schemas
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )


class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


# Initialize LLM
llm = init_chat_model("openai:gpt-4o-mini")
llm_router = llm.with_structured_output(Router)


# Initialize memory store
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)


# Template for formatting few-shot examples
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""


def format_few_shot_examples(examples):
    """Format list of few-shot examples for the prompt."""
    if not examples:
        return "No previous examples available."
    
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)


def add_example_to_memory(store, user_id: str, email_data: dict, label: str):
    """Add an example email to episodic memory."""
    data = {
        "email": email_data,
        "label": label
    }
    store.put(
        ("email_assistant", user_id, "examples"), 
        str(uuid.uuid4()), 
        data
    )
    print(f"✅ Added example to memory with label: {label}")


# Define tools
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"


@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


# Create memory tools
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)

search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)


# Agent prompt with memory support
agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""


def create_prompt(state):
    """Create the prompt for the agent."""
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["agent_instructions"], 
                **profile
            )
        }
    ] + state['messages']


# Create response agent
tools = [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool
]

response_agent = create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,
    store=store
)


# Define triage router node with episodic memory
def triage_router(state: State, config, store) -> Command[
    Literal["response_agent", "__end__"]
]:
    """Route emails based on classification with few-shot learning."""
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    # Search for similar examples in episodic memory
    namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )
    examples = store.search(
        namespace, 
        query=str({"email": state['email_input']})
    ) 
    examples_text = format_few_shot_examples(examples)
    
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=examples_text
    )
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    
    return Command(goto=goto, update=update)


# Build the email agent graph
def build_email_agent():
    """Build and compile the email agent."""
    email_agent = StateGraph(State)
    email_agent = email_agent.add_node(triage_router)
    email_agent = email_agent.add_node("response_agent", response_agent)
    email_agent = email_agent.add_edge(START, "triage_router")
    return email_agent.compile(store=store)


def process_email(email_agent, email_input: dict, config: dict):
    """Process an email through the agent."""
    print(f"\n{'='*80}")
    print(f"Processing email from: {email_input['author']}")
    print(f"Subject: {email_input['subject']}")
    print(f"{'='*80}\n")
    
    response = email_agent.invoke(
        {"email_input": email_input},
        config=config
    )
    
    if response.get("messages"):
        print("\n--- Agent Response ---")
        for m in response["messages"]:
            m.pretty_print()
    
    return response


def main():
    """Main execution function."""
    print("🚀 Starting Email Assistant with Episodic Memory\n")
    
    # Build the agent
    email_agent = build_email_agent()
    
    print("="*80)
    print("STEP 1: Adding Training Examples to Episodic Memory")
    print("="*80 + "\n")
    
    # Add training examples for user 'lance'
    example_1 = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
    }
    add_example_to_memory(store, "lance", example_1, "respond")
    
    example_2 = {
        "author": "Sarah Chen <sarah.chen@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Update: Backend API Changes Deployed to Staging",
        "email_thread": """Hi John,
    
Just wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:

- Implemented JWT refresh token rotation
- Added rate limiting for login attempts
- Updated API documentation with new endpoints

All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*

No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.

Best regards,
Sarah
""",
    }
    add_example_to_memory(store, "lance", example_2, "notify")
    
    print("\n" + "="*80)
    print("STEP 2: Testing with Harrison (No Examples)")
    print("="*80 + "\n")
    
    # Test with Harrison (no examples yet) - should respond
    config_harrison = {"configurable": {"langgraph_user_id": "harrison"}}
    
    email_spam = {
        "author": "Tom Jones <tom.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John - want to buy documentation?""",
    }
    
    print("🔬 Testing spam email (no examples in memory yet):")
    process_email(email_agent, email_spam, config_harrison)
    
    print("\n" + "="*80)
    print("STEP 3: Adding Example to Harrison's Memory")
    print("="*80 + "\n")
    
    # Add example to Harrison's memory
    add_example_to_memory(store, "harrison", email_spam, "ignore")
    
    print("\n" + "="*80)
    print("STEP 4: Re-testing Same Email (With Example)")
    print("="*80 + "\n")
    
    # Try again - should now ignore
    print("🔬 Re-testing same spam email (with example in memory):")
    process_email(email_agent, email_spam, config_harrison)
    
    print("\n" + "="*80)
    print("STEP 5: Testing Similar Email (Slight Variation)")
    print("="*80 + "\n")
    
    # Test with slight variation - should still ignore due to similarity
    email_spam_variation = {
        "author": "Jim Jones <jim.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John - want to buy documentation?????""",
    }
    
    print("🔬 Testing similar spam email:")
    process_email(email_agent, email_spam_variation, config_harrison)
    
    print("\n" + "="*80)
    print("STEP 6: Testing with Different User (Andrew)")
    print("="*80 + "\n")
    
    # Test with Andrew (no examples) - should respond
    config_andrew = {"configurable": {"langgraph_user_id": "andrew"}}
    
    print("🔬 Testing same email with different user (no examples):")
    process_email(email_agent, email_spam_variation, config_andrew)
    
    # Display memory information
    print("\n\n" + "="*80)
    print("Memory Store Summary")
    print("="*80)
    print("\nNamespaces in store:")
    namespaces = store.list_namespaces()
    for ns in namespaces:
        print(f"  - {ns}")
    
    print("\n✅ Email Assistant Demo Complete!")
    print("\nKey Takeaways:")
    print("1. Episodic memory allows the assistant to learn from examples")
    print("2. Few-shot learning improves classification accuracy")
    print("3. Memory is user-specific (harrison vs andrew)")
    print("4. Similar emails are handled consistently due to semantic search")


if __name__ == "__main__":
    main()
