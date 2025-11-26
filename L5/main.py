"""
Lesson 5: Email Assistant with Semantic + Episodic + Procedural Memory

This script implements an email assistant that:
- Classifies incoming messages (respond, ignore, notify)
- Drafts responses
- Schedules meetings
- Uses semantic memory to remember details from previous emails
- Uses episodic memory (few-shot examples) to improve classification accuracy
- Uses procedural memory to dynamically update its own instructions based on feedback
"""

import os
import json
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

# Prompt instructions (initial defaults)
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


def create_prompt(state, config, store):
    """Create the prompt for the agent - retrieves instructions from procedural memory."""
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )
    result = store.get(namespace, "agent_instructions")
    if result is None:
        store.put(
            namespace, 
            "agent_instructions", 
            {"prompt": prompt_instructions["agent_instructions"]}
        )
        prompt = prompt_instructions["agent_instructions"]
    else:
        prompt = result.value['prompt']
    
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt, 
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


# Define triage router node with episodic memory and procedural memory
def triage_router(state: State, config, store) -> Command[
    Literal["response_agent", "__end__"]
]:
    """Route emails based on classification with few-shot learning and dynamic instructions."""
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
    
    # Get triage rules from procedural memory (or use defaults)
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )

    result = store.get(namespace, "triage_ignore")
    if result is None:
        store.put(
            namespace, 
            "triage_ignore", 
            {"prompt": prompt_instructions["triage_rules"]["ignore"]}
        )
        ignore_prompt = prompt_instructions["triage_rules"]["ignore"]
    else:
        ignore_prompt = result.value['prompt']

    result = store.get(namespace, "triage_notify")
    if result is None:
        store.put(
            namespace, 
            "triage_notify", 
            {"prompt": prompt_instructions["triage_rules"]["notify"]}
        )
        notify_prompt = prompt_instructions["triage_rules"]["notify"]
    else:
        notify_prompt = result.value['prompt']

    result = store.get(namespace, "triage_respond")
    if result is None:
        store.put(
            namespace, 
            "triage_respond", 
            {"prompt": prompt_instructions["triage_rules"]["respond"]}
        )
        respond_prompt = prompt_instructions["triage_rules"]["respond"]
    else:
        respond_prompt = result.value['prompt']
    
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=ignore_prompt,
        triage_notify=notify_prompt,
        triage_email=respond_prompt,
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


def update_prompts_with_feedback(store, user_id: str, feedback: str, feedback_type: str = "agent"):
    """Update procedural memory prompts based on feedback - manual version without LLM optimizer."""
    print(f"\n{'='*80}")
    print("Updating Procedural Memory with Feedback")
    print(f"{'='*80}")
    print(f"Feedback: {feedback}")
    print(f"Type: {feedback_type}\n")
    
    # Manual prompt updates based on feedback type
    if feedback_type == "agent":
        # Update agent instructions
        current = store.get((user_id,), "agent_instructions").value['prompt']
        updated = f"{current} {feedback}"
        store.put((user_id,), "agent_instructions", {"prompt": updated})
        print(f"✅ Updated agent_instructions")
        print(f"   Old: {current}")
        print(f"   New: {updated}\n")
        
    elif feedback_type == "ignore":
        # Update triage ignore rules
        current = store.get((user_id,), "triage_ignore").value['prompt']
        # Extract the person/topic to ignore from feedback
        if "from" in feedback.lower():
            # e.g., "Ignore emails from Alice Jones"
            updated = f"{current}, emails from Alice Jones"
        else:
            updated = f"{current}, {feedback}"
        store.put((user_id,), "triage_ignore", {"prompt": updated})
        print(f"✅ Updated triage_ignore")
        print(f"   Old: {current}")
        print(f"   New: {updated}\n")
        
    elif feedback_type == "notify":
        # Update triage notify rules
        current = store.get((user_id,), "triage_notify").value['prompt']
        updated = f"{current}, {feedback}"
        store.put((user_id,), "triage_notify", {"prompt": updated})
        print(f"✅ Updated triage_notify")
        print(f"   Old: {current}")
        print(f"   New: {updated}\n")
        
    elif feedback_type == "respond":
        # Update triage respond rules
        current = store.get((user_id,), "triage_respond").value['prompt']
        updated = f"{current}, {feedback}"
        store.put((user_id,), "triage_respond", {"prompt": updated})
        print(f"✅ Updated triage_respond")
        print(f"   Old: {current}")
        print(f"   New: {updated}\n")
    
    print("✅ Procedural memory updated successfully!")


def display_memory_state(store, user_id: str):
    """Display the current state of procedural memory."""
    print(f"\n{'='*80}")
    print("Current Procedural Memory State")
    print(f"{'='*80}\n")
    
    agent_instructions = store.get((user_id,), "agent_instructions")
    if agent_instructions:
        print(f"Agent Instructions:\n  {agent_instructions.value['prompt']}\n")
    
    triage_ignore = store.get((user_id,), "triage_ignore")
    if triage_ignore:
        print(f"Triage Ignore Rules:\n  {triage_ignore.value['prompt']}\n")
    
    triage_notify = store.get((user_id,), "triage_notify")
    if triage_notify:
        print(f"Triage Notify Rules:\n  {triage_notify.value['prompt']}\n")
    
    triage_respond = store.get((user_id,), "triage_respond")
    if triage_respond:
        print(f"Triage Respond Rules:\n  {triage_respond.value['prompt']}\n")


def main():
    """Main execution function."""
    print("🚀 Starting Email Assistant with Procedural Memory\n")
    
    # Build the agent
    email_agent = build_email_agent()
    config = {"configurable": {"langgraph_user_id": "lance"}}
    
    print("="*80)
    print("STEP 1: Initial Email Processing (Default Instructions)")
    print("="*80)
    
    # Process initial email
    email_urgent = {
        "author": "Alice Jones <alice.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why""",
    }
    
    response_1 = process_email(email_agent, email_urgent, config)
    
    # Display current memory state
    display_memory_state(store, "lance")
    
    print("\n" + "="*80)
    print("STEP 2: Update Instructions with Feedback")
    print("="*80)
    
    # Update instructions based on feedback
    feedback_1 = "Always sign your emails `John Doe`"
    update_prompts_with_feedback(store, "lance", feedback_1, feedback_type="agent")
    
    # Display updated memory state
    display_memory_state(store, "lance")
    
    print("\n" + "="*80)
    print("STEP 3: Re-process Email with Updated Instructions")
    print("="*80)
    
    # Process same email again with updated instructions
    response_2 = process_email(email_agent, email_urgent, config)
    
    print("\n" + "="*80)
    print("STEP 4: Update Triage Rules with Feedback")
    print("="*80)
    
    # Update triage rules to ignore emails from Alice Jones
    feedback_2 = "Ignore any emails from Alice Jones"
    update_prompts_with_feedback(store, "lance", feedback_2, feedback_type="ignore")
    
    # Display updated memory state
    display_memory_state(store, "lance")
    
    print("\n" + "="*80)
    print("STEP 5: Process Email with Updated Triage Rules")
    print("="*80)
    
    # Process same email again - should now be ignored
    response_3 = process_email(email_agent, email_urgent, config)
    
    print("\n" + "="*80)
    print("Memory Store Summary")
    print("="*80)
    print("\nNamespaces in store:")
    namespaces = store.list_namespaces()
    for ns in namespaces:
        print(f"  - {ns}")
    
    print("\n✅ Email Assistant Demo Complete!")
    print("\nKey Takeaways:")
    print("1. Procedural memory allows the agent to update its own instructions")
    print("2. Feedback is automatically translated into prompt updates")
    print("3. Both agent behavior and triage rules can be dynamically modified")
    print("4. The agent learns and adapts based on user preferences")


if __name__ == "__main__":
    main()
