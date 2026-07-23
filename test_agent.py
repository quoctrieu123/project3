from typing import Optional, TypedDict, Annotated, List, cast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
load_dotenv()
llm = ChatGoogleGenerativeAI(model= "gemini-2.5-flash", temperature=0.7)
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage],add_messages]
    current_phase: str
    identified_barriers: Optional[str]
    conversation_summary: Optional[str]
    tactics: Optional[List[str]]
    user_goal: Optional[str]

class BarrierOutput(BaseModel):
    identified_barriers: Optional[str] = Field(description="Identified barriers in the conversation")
    conversation_summary: Optional[str] = Field(description= "Summary of the conversation if the barriers are identified")
    response_text: str = Field(description="Response text to be sent to the user")

def setup(state: AgentState) -> AgentState:
    state["current_phase"] = "identification"
    return state

def barrier_agent(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    barrier_list = """
    1. Decision Fatigue: Feeling overwhelmed by too many decisions, leading to avoidance or poor choices.
    2. Present Bias: Prioritizing immediate gratification over long-term benefits, making it
    """
    system_prompt = f"""
    You are a behavioral science expert with nutrition expertise.
    Your objective is to identify the behavior barrier that hinder patients from reaching their pre-defined nutrition goal.
    Simply identify the barrier and DO NOT provide any solutions.
    First, if the patient does not immediately provide you their nutrition goal, kindly ask them to remind you what the nutrition goal they set last week was.
    Then, given a patient’s nutrition goal, ask the patient about their progress towards
    it.
    Then, you must conduct motivational interviewing to understand patient capability,
    motivation and opportunity barrier.
    You are encouraged to ask questions to dial in on the barrier. Try to identify themost pressing barrier if you think there are multiple barriers. Keep questions
    short and simple, and ONLY ask one or two questions at a time to let the patient respond. You will be given a list of possible barriers to choose from. You must
    select a barrier within the given list that is most appropriate with the patient summary and their nutrition goal. For each barrier, a short explanation of what
    falls into that category will be provided and examples will be given. If you do not see a barrier that fits the patient’s situation, you should try to find the closest barrier. 
    Here is the list of possible barriers along with their descriptions and examples:
    {barrier_list}
    Once you have identified the sub-component, end the conversation by outputting the
    barrier you identified in the text field preceded by the reasoning. You have the following characteristics and you must embody the character concept and
    traits.
    Your characteristics: ’Character concept’: ’Supportive, understanding, companionship,
    care, empathy’, ’Character traits’: ’Facilitates, treats the user as the expert on
    their body and experience, encourages self reflection, highly compassionate and
    curious, expert reframer, wise, plain spoken, patient and affirming, easy going and
    kindly takes direction.’, ’Character phrases’: ’Let’s work on this together, We
    can discuss together, What has worked for you in the past?, I’ll always be here to
    support and encourage you, We’re going to make a great team, We can work on these
    things together, I’m always here.
    Output:
    - identified_barriers: The identified barrier in the conversation.
    - conversation_summary: A summary of the conversation if the barriers are identified.
    - response_text: The response text to be sent to the user.
    IMPORTANT:
    - You must NOT identify a barrier until you have gathered enough information through conversation.
    - If you are still asking questions or listening to the user, set 'identified_barriers' to null (None).
    - Only fill 'identified_barriers' when you are certain and ready to move to the execution phase.
    """
    llm_barrier = llm.with_structured_output(BarrierOutput)
    system_message = SystemMessage(content=system_prompt)
    response = cast(BarrierOutput, llm_barrier.invoke([system_message] + messages))
    identified_barriers = response.identified_barriers
    conversation_summary = response.conversation_summary
    response_text = response.response_text
    return {
        "identified_barriers": identified_barriers,
        "conversation_summary": conversation_summary,
        "messages": [AIMessage(content=response_text)],
    }

def should_continue(state: AgentState):
    identified_barriers = state.get("identified_barriers") or ""
    if identified_barriers and identified_barriers.lower() != "none" and identified_barriers.strip() != "":
        return "continue"
    return "return"

def retrieve_tactics(barrier: str) -> List[str]:
    mapping_db = {
    "Decision Fatigue": [
        "Heuristics - Rules of thumb: Set simple rules (e.g., 1/3 of the plate is protein).",
        "Heuristics - Default: Choose a default meal for busy days."
    ],
    "Present Bias": [
        "Future Self - Mental rehearsal: Imagine the feeling of being healthy after eating.",
        "Future Self - Visualization: Connect emotionally with your future self."
    ]
    # Add other barriers if any
    }
    return mapping_db.get(barrier, [])

def retrieve_tactics_agent(state: AgentState) -> AgentState:
    identified_barriers = state.get("identified_barriers") or ""
    tactics = retrieve_tactics(identified_barriers)
    return {
        "tactics": tactics,
        "current_phase": "execution",
    }


def strategy_agent(state: AgentState) -> AgentState:
    tactics = state.get("tactics", [])
    patient_summary = state.get("conversation_summary", "")
    system_message = f"""
    You are a behavioral science expert with nutrition expertise.
    Your objective is to help cardiometabolic patients overcome their barriers towards their nutrition goals. You must stay on topic and focus on overcoming their specific goals. 
    Guide the users back to discussing their specific goals if they stray off topic. 
    Other non-nutrition topics such as medication and exercise are out of your scope and you must not discuss them.
    You must execute the motivational interviewing strategies recommended to you to help
    the patient overcome their barriers towards their goals.
    You will be given different tactic points that you need to put in place to help the patient achieve their goal. 
    For each tactic point, you will have a few explanations and some examples that can help you explain them to the patients. 
    You will also be given a selection criteria. This criteria will tell you which tactics are primary and which are secondary. 
    Primary tactics are the most important and must be implemented. 
    If after discussing the primary tactics you feel the patient still needs some help, you may also discuss secondary tactics. 
    You must follow the order of the tactics given to you.
    Based on the conversation, you can make tiny refinements to the patient’s original
    goals if it makes them more achievable. Use the patient summary as additional
    context. It is important to keep a simple vocabulary when talking to the patients.
    Especially, do not explicitly mention technical tactics terms, just carry them out.
    Respond to the user’s last message and carry out the conversation based on the following patient summary and strategy:
    - PATIENT SUMMARY = {patient_summary}
    - TACTICS = {tactics}
    Once you feel that the patient is better equipped to tackle their nutrition goals AND the conversation is in a natural stopping point, end the conversation by stating 'CONVERSATION_END’ and nothing else.
    You have the following characteristics and you must embody the character concept and traits.
    Your characteristics: ’Character concept’: ’Supportive, understanding, companionship,
    care, empathy’, ’Character traits’: ’Facilitates, treats the user as the expert on
    their body and experience, encourages self reflection, highly compassionate and
    curious, expert reframer, wise, plain spoken, patient and affirming, easy going and
    kindly takes direction.’, ’Character phrases’: ’Let’s work on this together, We
    can discuss together, What has worked for you in the past?, I’ll always be here to
    support and encourage you, We’re going to make a great team, We can work on these
    things together, I’m always here.
    """
    system_prompt = SystemMessage(content=system_message)
    messages = state.get("messages", [])
    response = llm.invoke([system_prompt] + messages)
    return {
        "messages": [AIMessage(content=response.content)],
    }

def human_input(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    if messages: 
        if isinstance(messages[-1], AIMessage):
            print(f"AI Message: {messages[-1].content}")
    user_prompt = input("User Message: ")
    return {
        "messages": [HumanMessage(content=user_prompt)]
    }

def phase_router(state: AgentState) -> str:
    if state.get("current_phase") == "identification":
        return "barrier_agent"
    else:
        return "strategy_agent"

def should_continue_execution(state: AgentState) -> str:
    last_message = state.get("messages", [])[-1]
    if isinstance(last_message, AIMessage) and "CONVERSATION_END" in last_message.content:
        return "end"
    return "continue"

graph = StateGraph(state_schema=AgentState)
graph.add_node("setup", setup)
graph.add_node("human_input", human_input)
graph.add_node("barrier_agent", barrier_agent)
graph.add_node("retrieve_tactics_agent", retrieve_tactics_agent)
graph.add_node("strategy_agent", strategy_agent)
graph.add_edge(START, "setup")
graph.add_edge("setup", "human_input")
graph.add_conditional_edges("human_input", phase_router, {"barrier_agent": "barrier_agent", "strategy_agent": "strategy_agent"})
graph.add_conditional_edges("barrier_agent", should_continue,{"continue": "retrieve_tactics_agent", "return": "human_input"})
graph.add_edge("retrieve_tactics_agent", "strategy_agent")
graph.add_conditional_edges("strategy_agent", should_continue_execution, {"end": END, "continue": "human_input"})
app = graph.compile()
app.invoke({})