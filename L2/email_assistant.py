#!/usr/bin/env python
"""
Email Assistant - Baseline Implementation
A LangGraph agent that classifies incoming emails and drafts responses.

This script builds an email assistant that:
- Classifies incoming messages (respond, ignore, notify)
- Drafts responses using AI
- Schedules meetings
"""

import os
from typing import Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# Import local modules
from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt
from schemas import Router, State


# Load environment variables
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}


# ============================================================================
# Tools
# ============================================================================

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


# ============================================================================
# LLM Setup
# ============================================================================

llm = init_chat_model("openai:gpt-4o-mini")
llm_router = llm.with_structured_output(Router)

tools = [write_email, schedule_meeting, check_calendar_availability]


# ============================================================================
# Agent Functions
# ============================================================================

def create_prompt(state):
    """Create the prompt for the main agent."""
    return [
        {
            "role": "system", 
            "content": agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile
            )
        }
    ] + state['messages']


def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """
    Triage incoming emails and route them appropriately.
    
    Returns a Command that either:
    - Routes to response_agent if email needs a response
    - Routes to END if email should be ignored or just notified
    """
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
        print(f"   Reasoning: {result.reasoning}")
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
        print(f"   Reasoning: {result.reasoning}")
        update = None
        goto = END
    elif result.classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")
        print(f"   Reasoning: {result.reasoning}")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    
    return Command(goto=goto, update=update)


# ============================================================================
# Build the Email Agent
# ============================================================================

def build_email_agent():
    """Build and compile the email agent graph."""
    # Create the response agent
    response_agent = create_react_agent(
        "openai:gpt-4o",
        tools=tools,
        prompt=create_prompt,
    )
    
    # Build the graph
    email_agent = StateGraph(State)
    email_agent = email_agent.add_node(triage_router)
    email_agent = email_agent.add_node("response_agent", response_agent)
    email_agent = email_agent.add_edge(START, "triage_router")
    email_agent = email_agent.compile()
    
    return email_agent


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Main function demonstrating the email assistant."""
    
    # Build the agent
    email_agent = build_email_agent()
    
    # Example 1: Marketing email (should be ignored)
    print("\n" + "="*80)
    print("EXAMPLE 1: Marketing Email")
    print("="*80)
    
    email_input_1 = {
        "author": "Marketing Team <marketing@amazingdeals.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
        "email_thread": """Dear Valued Developer,

Don't miss out on this INCREDIBLE opportunity! 

🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite! 

✨ FEATURES:
- Revolutionary AI-powered code completion
- Cloud-based development environment
- 24/7 customer support
- And much more!

💰 Regular Price: $999/month
🎉 YOUR SPECIAL PRICE: Just $199/month!

🕒 Hurry! This offer expires in:
24 HOURS ONLY!

Click here to claim your discount: https://amazingdeals.com/special-offer

Best regards,
Marketing Team
---
To unsubscribe, click here
""",
    }
    
    response_1 = email_agent.invoke({"email_input": email_input_1})
    print(f"\nResult: {len(response_1.get('messages', []))} messages in response")
    
    # Example 2: Direct question from team member (should respond)
    print("\n" + "="*80)
    print("EXAMPLE 2: Team Member Question")
    print("="*80)
    
    email_input_2 = {
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
    
    response_2 = email_agent.invoke({"email_input": email_input_2})
    
    print("\n" + "-"*80)
    print("MESSAGES EXCHANGED:")
    print("-"*80)
    for i, msg in enumerate(response_2["messages"], 1):
        print(f"\nMessage {i}:")
        msg.pretty_print()
    
    print("\n" + "="*80)
    print("Email Assistant Demo Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
