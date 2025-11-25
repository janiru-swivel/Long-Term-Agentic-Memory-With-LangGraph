"""
Lesson 3: Email Assistant with Semantic Memory

This script implements an email assistant that:
- Classifies incoming messages (respond, ignore, notify)
- Drafts responses
- Schedules meetings
- Remembers details from previous emails using semantic memory
"""

import os
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


# Define tools
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"


@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"


@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


# Initialize memory store
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

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
    "anthropic:claude-3-5-sonnet-latest",
    tools=tools,
    prompt=create_prompt,
    store=store
)


# Define triage router node
def triage_router(state: State) -> Command[
    Literal["response_agent", "__end__"]
]:
    """Route emails based on classification."""
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None
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
    
    print("\n--- Agent Response ---")
    for m in response["messages"]:
        m.pretty_print()
    
    return response


def main():
    """Main execution function."""
    print("🚀 Starting Email Assistant with Semantic Memory\n")
    
    # Build the agent
    email_agent = build_email_agent()
    
    # Configuration for user
    config = {"configurable": {"langgraph_user_id": "lance"}}
    
    # Example 1: Initial email about API documentation
    email_input_1 = {
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
    
    response_1 = process_email(email_agent, email_input_1, config)
    
    # Example 2: Follow-up email
    print("\n\n" + "="*80)
    print("Processing follow-up email...")
    print("="*80 + "\n")
    
    email_input_2 = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Follow up",
        "email_thread": """Hi John,

Any update on my previous ask?""",
    }
    
    response_2 = process_email(email_agent, email_input_2, config)
    
    # Display memory information
    print("\n\n" + "="*80)
    print("Memory Store Information")
    print("="*80)
    print("\nNamespaces in store:")
    print(store.list_namespaces())
    
    print("\nSearching memory for 'Alice':")
    results = store.search(('email_assistant', 'lance', 'collection'), query="Alice")
    for result in results:
        print(f"- {result}")
    
    print("\n✅ Email Assistant Demo Complete!")


if __name__ == "__main__":
    main()
