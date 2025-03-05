import os
import discord
import logging
import asyncio
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")

# Set up logging to console
logger = logging.getLogger("therapy")
logging.basicConfig(level=logging.INFO)

# Define the state schema for our graph.
class TherapyState(TypedDict):
    messages: List[Dict[str, Any]]
    intake_data: Dict[str, str]
    initial_context: str
    intake_phase: bool
    intake_topics_covered: int
    asked_topics: List[str]
    delete_confirmation: bool
    user_id: str
    channel_id: str
    intent: str
    followup_count: int

class TherapistAgent:
    def __init__(self, channel_id, user_id=None):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
        self.channel_id = channel_id
        self.user_id = user_id
        self.channel = None
        self.delete_confirmation = False
        
        # Set the minimum number of follow-up questions to be asked during intake.
        self.MIN_FOLLOWUPS = 3

        # System prompt to enforce a warm, adaptive, and compassionate tone.
        self.system_prompt = SystemMessage(content=(
            "You are a deeply compassionate therapist. Listen attentively, express genuine empathy, "
            "and ask adaptive follow-up questions related to the client's situation. "
            "Avoid repeating the same questions and ensure each response feels warm and supportive."
        ))
        self.prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        self.chat_history = []
        
        # This list tracks topics that have already been used as the basis for a follow-up question.
        self.intake_topics = []

        # Build the LangGraph workflow.
        self.workflow = self.build_graph()

        # Additional attributes for state.
        self.initial_context = ""
        self.intake_data = {}

    def generate_standalone_followup(self, state: TherapyState) -> str:
        """
        Generate a self-contained follow-up question that captures the latest conversation context.
        This method is invoked when no new undiscussed topic is found and ensures that the question
        remains contextually coherent and compassionate.
        """
        conversation = "\n".join([msg["content"] for msg in state["messages"]])
        prompt = (
            "Based on the following conversation, generate a self-contained follow-up question "
            "that captures the client's latest input in a warm and compassionate manner. "
            "If no specific theme is evident, ask a gentle generic follow-up question:\n\n" +
            conversation
        )
        response = self.llm.invoke([
            SystemMessage(content="You are a warm, compassionate therapist who listens attentively."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()

    def build_graph(self):
        builder = StateGraph(TherapyState)

        # Add nodes that update the therapy state.
        builder.add_node("analyze_intent", self.analyze_intent)
        builder.add_node("process_intake", self.process_intake)
        builder.add_node("complete_intake", self.complete_intake)
        builder.add_node("handle_crisis", self.handle_crisis_node)
        builder.add_node("handle_delete_request", self.handle_delete_request_node)
        builder.add_node("normal_therapy", self.normal_therapy)

        # Store MIN_FOLLOWUPS locally so it can be used inside our nested function.
        MIN_FOLLOWUPS = self.MIN_FOLLOWUPS

        def route_analyze(state: TherapyState) -> str:
            if state["intent"] == "CRISIS":
                return "handle_crisis"
            elif state["intent"] == "DELETE_REQUEST":
                return "handle_delete_request"
            elif state["intent"] == "NEUTRAL":
                if state["intake_phase"]:
                    # Force additional follow-ups until the minimum has been reached.
                    if state.get("followup_count", 0) < MIN_FOLLOWUPS:
                        return "process_intake"
                    # Otherwise, if we have collected enough entries or no new topics are extracted, complete intake.
                    elif state["intake_topics_covered"] >= 10 or len(self.extract_new_topics(state)) == 0:
                        return "complete_intake"
                    else:
                        return "process_intake"
                else:
                    return "normal_therapy"
            else:
                return END

        builder.add_conditional_edges("analyze_intent", route_analyze)
        # Also check in process_intake whether we need to complete intake.
        builder.add_conditional_edges(
            "process_intake",
            lambda state: "complete_intake" if (state["intake_topics_covered"] >= 10 or 
                                                   (state.get("followup_count", 0) >= MIN_FOLLOWUPS and len(self.extract_new_topics(state)) == 0))
                                                 else END
        )

        builder.add_edge("handle_crisis", END)
        builder.add_edge("handle_delete_request", END)
        builder.add_edge("complete_intake", END)
        builder.add_edge("normal_therapy", END)

        builder.set_entry_point("analyze_intent")
        return builder.compile()

    def extract_new_topics(self, state: TherapyState) -> List[str]:
        """
        Dynamically extract new topics from the entire conversation history.
        The LLM is prompted to list up to two topics about the client's emotional or personal experience
        that have not already been discussed. If none are found, it returns an empty list.
        """
        conversation = "\n".join([str(msg["content"]) for msg in state["messages"]])
        prompt = (
            "Based on the following conversation, please suggest up to two specific topics "
            "about the client's emotional or personal experience that have not yet been discussed. "
            "If no additional topics can be identified, respond with 'None'.\n\n" +
            conversation
        )
        result = self.llm.invoke([
            SystemMessage(content="You are an expert therapist summarizing conversation topics."),
            HumanMessage(content=prompt)
        ]).content.strip()
        if result.lower().startswith("none"):
            return []
        topics = [t.strip() for t in result.split(",") if t.strip()]
        logger.info(f"[EXTRACT TOPICS] New topics: {topics}")
        return topics

    def analyze_intent(self, state: TherapyState) -> TherapyState:
        messages = state["messages"]
        user_message = messages[-1]["content"] if messages else ""
        intent_prompt = (
            "Analyze the following message for its intent:\n"
            "1. CRISIS - if there are expressions of self-harm, suicide, or extreme hopelessness.\n"
            "2. DELETE_REQUEST - if there are requests to delete or terminate the conversation.\n"
            "3. NEUTRAL - for all other messages.\n\n"
            "Respond ONLY with one of: CRISIS, DELETE_REQUEST, or NEUTRAL."
        )
        response = self.llm.invoke([
            SystemMessage(content=intent_prompt),
            HumanMessage(content=f"Message: {user_message}")
        ])
        if "CRISIS" in response.content:
            intent = "CRISIS"
        elif "DELETE" in response.content:
            intent = "DELETE_REQUEST"
        else:
            intent = "NEUTRAL"
        logger.info(f"[ANALYZE INTENT] {user_message[:50]}... -> {intent}")
        return {**state, "intent": intent}

    def process_intake(self, state: TherapyState) -> TherapyState:
        """
        Processes the user's message during the intake phase by:
        1. Recording the message as a new entry.
        2. Ensuring that a minimum of three follow-up questions are asked.
        3. Generating contextually relevant follow-ups either based on extracted topics or via a standalone prompt.
        """
        messages = state["messages"]
        user_message = messages[-1]["content"] if messages else ""
        intake_data = state["intake_data"]

        logger.info(f"[PROCESS INTAKE] Received message: {user_message}")

        # Record the user's message as a new entry.
        entry_key = f"Entry {len(intake_data) + 1}"
        intake_data[entry_key] = user_message

        # Retrieve follow-up count and topics already used.
        followup_count = state.get("followup_count", 0)
        asked_topics = state.get("asked_topics", [])
        MIN_FOLLOWUPS = self.MIN_FOLLOWUPS

        if followup_count < MIN_FOLLOWUPS:
            # Try to extract new topics from the conversation context.
            extracted_topics = self.extract_new_topics(state)
            if extracted_topics:
                # Use only topics not yet asked.
                candidate_topics = [t for t in extracted_topics if t not in asked_topics]
                if candidate_topics:
                    chosen_topic = candidate_topics[0]
                    prompt = (
                        f"Considering the client's recent messages, generate a thoughtful, open-ended follow-up question "
                        f"about '{chosen_topic}' that invites the client to elaborate on their feelings and context."
                    )
                    followup_question = self.llm.invoke([
                        SystemMessage(content="You are a warm and empathetic therapist."),
                        HumanMessage(content=prompt)
                    ]).content.strip()
                    asked_topics.append(chosen_topic)
                else:
                    # Fallback: generate a standalone follow-up.
                    followup_question = self.generate_standalone_followup(state)
            else:
                # Fallback when no topics are extracted.
                followup_question = self.generate_standalone_followup(state)
            followup_count += 1
        else:
            # After reaching the minimum follow-ups, provide a gentle final check-in prompt.
            followup_question = (
                "I appreciate your openness today. Could you share any additional thoughts or feelings you might have? "
                "I’m here to listen and support you through this process."
            )

        updated_state = {
            **state,
            "intake_data": intake_data,
            "asked_topics": asked_topics,
            "followup_count": followup_count,
            "intake_topics_covered": len(intake_data),
            "messages": messages + [{"role": "assistant", "content": followup_question}]
        }

        logger.info(f"[PROCESS INTAKE] Total entries: {len(intake_data)}; Follow-up count: {followup_count}")
        return updated_state

    def complete_intake(self, state: TherapyState) -> TherapyState:
        """
        Summarizes all the intake entries and ends the intake phase.
        """
        intake_data = state["intake_data"]
        logger.info(f"[INTAKE COMPLETE] {len(intake_data)} entries collected for user {state['user_id']}.")
        summary_lines = [f"{key}: {value}" for key, value in intake_data.items()]
        summary = "\n".join(summary_lines)
        final_message = (
            "Thank you for sharing your experiences with such openness. Here is a summary of what you've shared:\n\n"
            f"{summary}\n\n"
            "We'll now shift our focus to how I can best support you moving forward."
        )
        updated_state = {
            **state,
            "intake_phase": False,
            "messages": state["messages"] + [{"role": "assistant", "content": final_message}]
        }
        return updated_state

    def handle_crisis_node(self, state: TherapyState) -> TherapyState:
        crisis_response = (
            "I understand you're experiencing deep pain right now. If you feel unsafe or have thoughts of harming yourself, "
            "please immediately call emergency services (e.g., 988 if in the US) or seek help from someone you trust. "
            "You matter, and I'm here to support you."
        )
        updated_state = {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": crisis_response}]
        }
        return updated_state

    def handle_delete_request_node(self, state: TherapyState) -> TherapyState:
        delete_response = (
            "If you'd like to end this session and delete our conversation, "
            "please type 'YES, END SESSION'. Otherwise, I'm here to continue supporting you."
        )
        updated_state = {
            **state,
            "delete_confirmation": True,
            "messages": state["messages"] + [{"role": "assistant", "content": delete_response}]
        }
        return updated_state

    def normal_therapy(self, state: TherapyState) -> TherapyState:
        response_message = (
            "Thank you for sharing. I hear you, and I'm here to support you as we continue our conversation."
        )
        updated_state = {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": response_message}]
        }
        return updated_state

    def get_response(self, user_message: str) -> str:
        """
        Processes the user's message by updating the state and executing the workflow.
        Returns the assistant's latest response.
        """
        if not hasattr(self, 'state'):
            self.state = TherapyState(
                messages=[{"role": "user", "content": user_message}],
                intake_data={},
                initial_context=self.initial_context,
                intake_phase=True,
                intake_topics_covered=0,
                asked_topics=[],
                followup_count=0,
                delete_confirmation=self.delete_confirmation,
                user_id=str(self.user_id),
                channel_id=str(self.channel_id),
                intent="NEUTRAL"
            )
        else:
            self.state = {
                **self.state,
                "messages": self.state["messages"] + [{"role": "user", "content": user_message}]
            }
        self.state = self.workflow.invoke(self.state)
        last_message = self.state["messages"][-1]
        # Update our local copies.
        self.intake_data = self.state["intake_data"]
        self.delete_confirmation = self.state["delete_confirmation"]
        return last_message["content"] if last_message["role"] == "assistant" else "I'm processing your message."

    # Legacy methods for compatibility.
    def handle_crisis(self) -> str:
        return (
            "I understand you're in deep pain. If you feel unsafe, please reach out immediately or call 988 (if in the US). "
            "Remember, you're not alone—I am here to support you."
        )

    def request_delete_confirmation(self, *args, **kwargs) -> str:
        self.delete_confirmation = True
        return (
            "Would you like to end this session and delete our conversation? Type 'YES, END SESSION' to confirm, "
            "or continue chatting if not."
        )

    def delete_channel(self, *args, **kwargs) -> str:
        if self.channel and self.delete_confirmation:
            asyncio.create_task(self.channel.delete())
            return "Channel deletion initiated. Please take care."
        return "Deletion failed – no confirmation received."

    async def start_session(self, channel: discord.TextChannel, user: discord.Member, user_message: str):
        """
        Starts the session by sending a greeting through a private channel.
        """
        self.channel = channel
        logger.info(f"[INTAKE LOG] Started intake for user {user.id} with: {user_message[:50]}...")
        greeting_prompt = (
            f'You shared: "{user_message}"\n'
            "I truly appreciate your courage in opening up about your feelings. "
            "Could you tell me a bit more about the emotions you're experiencing right now? I'm here to listen wholeheartedly."
        )
        greeting = self.llm.invoke([
            SystemMessage(content="You are a warm, compassionate therapist who listens intently."),
            HumanMessage(content=greeting_prompt)
        ])
        await channel.send(greeting.content)
        await user.send(f"Private session started: {channel.mention}")
        self.state = TherapyState(
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": greeting.content}
            ],
            intake_data={"Presenting problem and symptoms": user_message},
            initial_context=user_message,
            intake_phase=True,
            intake_topics_covered=1,
            asked_topics=[],
            followup_count=0,
            delete_confirmation=False,
            user_id=str(user.id),
            channel_id=str(channel.id),
            intent="NEUTRAL"
        )
        self.intake_data = {"Presenting problem and symptoms": user_message}

    async def handle_message(self, message: discord.Message):
        """
        Handles incoming messages. If deletion confirmation is active and the user confirms,
        deletes the channel; otherwise, processes the message and responds.
        """
        if self.delete_confirmation and message.content.strip().upper() == "YES, END SESSION":
            await message.channel.send("Deleting channel... Thank you for your time.")
            await asyncio.sleep(2)
            await message.channel.delete()
        else:
            response = self.get_response(message.content)
            await message.channel.send(response)

    @staticmethod
    def generate_warning_response(user_message: str) -> str:
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
        sys_prompt = SystemMessage(
            content="You are a thoughtful moderator. Generate a clear, compassionate warning message explaining why harmful language is unacceptable and its negative effects. End with a clear statement that this is a warning."
        )
        human_msg = HumanMessage(content=f"User said: {user_message}")
        resp = llm.invoke([sys_prompt, human_msg])
        return resp.content

    @staticmethod
    async def create_private_channel(guild: discord.Guild, user: discord.Member, bot_user: discord.ClientUser) -> discord.TextChannel:
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(view_channel=False),
            user: discord.PermissionOverwrite(view_channel=True, send_messages=True),
            bot_user: discord.PermissionOverwrite(view_channel=True, send_messages=True),
        }
        channel = await guild.create_text_channel(
            name=f"therapy-{user.name}",
            overwrites=overwrites
        )
        return channel
