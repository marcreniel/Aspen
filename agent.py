import os
import discord
import logging
import asyncio
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
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

class TherapistAgent:
    def __init__(self, channel_id, user_id=None):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
        self.channel_id = channel_id
        self.user_id = user_id
        self.channel = None
        self.delete_confirmation = False

        # System prompt defines a warm, compassionate tone.
        self.system_prompt = SystemMessage(content=(
            "You are a deeply compassionate therapist. Listen actively, express genuine empathy, and ask adaptive questions that fit the user's situation. "
            "Avoid repeating the same phrasing and always speak in a warm, understanding manner."
        ))

        self.prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        self.chat_history = []

        # Set up tools for session termination and channel deletion.
        self.request_delete_confirmation_tool = Tool(
            name="request_delete_confirmation",
            func=self.request_delete_confirmation,
            description="Confirm session termination request"
        )
        self.delete_channel_tool = Tool(
            name="delete_channel",
            func=self.delete_channel,
            description="Delete therapy channel after confirmation"
        )
        self.agent = initialize_agent(
            [self.request_delete_confirmation_tool, self.delete_channel_tool],
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3
        )

        # Define 9 intake topics.
        self.intake_topics = [
            "Personal and family medical/mental health history",
            "Current life situation (work, relationships, stressors)",
            "Previous therapy experiences and coping mechanisms",
            "Substance use history",
            "Risk assessment (suicidal thoughts, self-harm)",
            "Treatment goals and expectations",
            "Cultural background and beliefs",
            "Current medications and medical conditions",
            "Support system and social relationships"
        ]
        self.initial_context = ""

        # Map topics to keywords for flexible matching.
        self.topic_keywords = {
            "Personal and family medical/mental health history": ["personal", "family", "health", "mental health", "history"],
            "Current life situation (work, relationships, stressors)": ["work", "relationship", "personal life", "stress", "stressor"],
            "Previous therapy experiences and coping mechanisms": ["therapy", "counseling", "coping", "experience"],
            "Substance use history": ["substance", "drug", "weed", "alcohol", "marijuana"],
            "Risk assessment (suicidal thoughts, self-harm)": ["suicidal", "self-harm", "harm myself", "risk"],
            "Treatment goals and expectations": ["treatment", "goal", "expectation", "hope", "future"],
            "Cultural background and beliefs": ["cultural", "tradition", "belief", "background"],
            "Current medications and medical conditions": ["medicine", "medications", "condition", "illness", "medical"],
            "Support system and social relationships": ["support", "friend", "family", "social", "relationship"]
        }

        # Custom adaptive follow-up questions per topic.
        self.custom_questions = {
            "Personal and family medical/mental health history": 
                "I understand it can be very personal. If you feel safe, could you share some details about your family's health or your own mental health journey?",
            "Current life situation (work, relationships, stressors)":
                "It sounds as though life might be feeling overwhelming. Can you tell me more about your work, relationships, or any current stressors?",
            "Previous therapy experiences and coping mechanisms":
                "Sometimes old patterns reveal much about what's effective. Have you ever sought therapy or found personal ways to cope with difficult times?",
            "Substance use history":
                "I know this can be sensitive—if you're comfortable sharing, how have substances figured into your way of managing stress?",
            "Risk assessment (suicidal thoughts, self-harm)":
                "This topic is really hard to discuss, but if you feel at ease, could you tell me if you've ever had thoughts that worried you, such as self-harm or feeling unsafe?",
            "Treatment goals and expectations":
                "Sometimes looking forward can help us see a way out. What hopes or goals do you have for yourself, even if they seem small?",
            "Cultural background and beliefs":
                "Our values and cultural background shape so much of our identity. Would you be willing to share a bit about your cultural roots or important beliefs?",
            "Current medications and medical conditions":
                "Sometimes what's occurring physically can affect how we feel emotionally. If you're comfortable, could you tell me about any medications you take or any conditions you face?",
            "Support system and social relationships":
                "A strong support system can make a huge difference. Could you describe your relationships with close friends or family, or who you rely on during tough times?"
        }

        # Build the LangGraph workflow.
        self.workflow = self.build_graph()

    def build_graph(self):
        builder = StateGraph(TherapyState)

        # Add nodes.
        builder.add_node("analyze_intent", self.analyze_intent)
        builder.add_node("process_intake", self.process_intake)
        builder.add_node("complete_intake", self.complete_intake)
        builder.add_node("handle_crisis", self.handle_crisis_node)
        builder.add_node("handle_delete_request", self.handle_delete_request_node)
        builder.add_node("normal_therapy", self.normal_therapy)

        # Routing function from analyze_intent.
        def route_analyze(state: TherapyState) -> str:
            if state["intent"] == "CRISIS":
                return "handle_crisis"
            elif state["intent"] == "DELETE_REQUEST":
                return "handle_delete_request"
            elif state["intent"] == "NEUTRAL":
                # If intake_phase is active and either 10 topics have been collected
                # or there are no remaining new topics to ask then complete intake.
                if state["intake_phase"]:
                    if state["intake_topics_covered"] >= 10 or len([t for t in self.intake_topics if t not in state.get("asked_topics", [])]) == 0:
                        return "complete_intake"
                    else:
                        return "process_intake"
                else:
                    return "normal_therapy"
            else:
                return END

        builder.add_conditional_edges("analyze_intent", route_analyze)

        # In process_intake, if 10 items are collected or no questions remain, transition to complete_intake.
        topics = self.intake_topics
        builder.add_conditional_edges(
            "process_intake",
            lambda state: "complete_intake" if (state["intake_topics_covered"] >= 10 or 
                                                  len([t for t in topics if t not in state.get("asked_topics", [])]) == 0)
                           else END
        )

        # Other nodes end the workflow.
        builder.add_edge("handle_crisis", END)
        builder.add_edge("handle_delete_request", END)
        builder.add_edge("complete_intake", END)
        builder.add_edge("normal_therapy", END)
        builder.set_entry_point("analyze_intent")
        return builder.compile()

    def analyze_intent(self, state: TherapyState) -> TherapyState:
        messages = state["messages"]
        user_message = messages[-1]["content"] if messages else ""
        intent_prompt = (
            "Analyze message for these intents:\n"
            "1. CRISIS - Expressions of self-harm/suicide/extreme hopelessness.\n"
            "2. DELETE_REQUEST - Clear channel management requests.\n"
            "3. NEUTRAL - Other messages.\n\n"
            "Respond ONLY with: CRISIS, DELETE_REQUEST, or NEUTRAL."
        )
        crisis_check = self.llm.invoke([
            SystemMessage(content=intent_prompt),
            HumanMessage(content=f"Message: {user_message}")
        ])
        if "CRISIS" in crisis_check.content:
            intent = "CRISIS"
        elif "DELETE" in crisis_check.content:
            intent = "DELETE_REQUEST"
        else:
            intent = "NEUTRAL"
        logger.info(f"[ANALYZE INTENT] Input: {user_message[:50]}... | Determined intent: {intent}")
        return {**state, "intent": intent}

    def process_intake(self, state: TherapyState) -> TherapyState:
        messages = state["messages"]
        user_message = messages[-1]["content"] if messages else ""
        intake_data = state["intake_data"]
        asked_topics = state.get("asked_topics", [])

        logger.info(f"[PROCESS INTAKE] Received message: {user_message}")
        analysis_result = self.llm.invoke([
            SystemMessage(content="You are a therapist analyzing client responses for relevant intake topics."),
            HumanMessage(content=f'Analyze: "{user_message}"')
        ]).content.lower()
        logger.info(f"[PROCESS INTAKE] Analysis result: {analysis_result}")

        # Check each topic using keyword matching.
        for topic, keywords in self.topic_keywords.items():
            if topic not in intake_data:
                for kw in keywords:
                    if kw in user_message.lower() or kw in analysis_result:
                        intake_data[topic] = user_message
                        logger.info(f"[INTAKE LOG] Captured info for '{topic}': {user_message[:50]}...")
                        print(f"[INTAKE LOG] Captured info for '{topic}': {user_message[:50]}...")
                        break

        intake_topics_covered = len(intake_data)
        # Select next topic that is missing and has not been asked.
        missing_topics = [topic for topic in self.intake_topics if topic not in intake_data and topic not in asked_topics]
        if missing_topics:
            next_topic = missing_topics[0]
            asked_topics.append(next_topic)
            follow_up_question = self.custom_questions.get(next_topic,
                f"If you're comfortable, could you share a bit more about your {next_topic.lower()}? I'm here to listen.")
        else:
            follow_up_question = (
                "I truly appreciate your openness. It seems we've discussed many aspects already; if there's anything else you'd like to share, I'm here to listen."
            )
        updated_state = {
            **state,
            "intake_data": intake_data,
            "intake_topics_covered": intake_topics_covered,
            "asked_topics": asked_topics,
            "messages": messages + [{"role": "assistant", "content": follow_up_question}]
        }
        logger.info(f"[PROCESS INTAKE] Total topics collected: {intake_topics_covered}")
        return updated_state

    def complete_intake(self, state: TherapyState) -> TherapyState:
        intake_data = state["intake_data"]
        logger.info(f"[INTAKE COMPLETE] Collected {len(intake_data)} topics for user {state['user_id']}.")
        print(f"[INTAKE COMPLETE] Collected {len(intake_data)} topics for user {state['user_id']}.")
        final_message = (
            "Thank you so much for sharing with me. I've gathered enough information to understand your situation better, "
            "and we'll now focus on supporting you moving forward."
        )
        updated_state = {
            **state,
            "intake_phase": False,
            "messages": state["messages"] + [{"role": "assistant", "content": final_message}]
        }
        return updated_state

    def handle_crisis_node(self, state: TherapyState) -> TherapyState:
        crisis_response = (
            "I understand you're experiencing deep pain right now. If you feel unsafe or at risk, please consider calling 988 (in the US) or seeking immediate help. "
            "Remember, you are not alone—I care about you, and I'm here to support you."
        )
        updated_state = {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": crisis_response}]
        }
        return updated_state

    def handle_delete_request_node(self, state: TherapyState) -> TherapyState:
        delete_response = (
            "It sounds like you might be overwhelmed. If you'd like to end this session and delete the channel, please type 'YES, END SESSION'. "
            "Otherwise, I'm here to keep supporting you."
        )
        updated_state = {
            **state,
            "delete_confirmation": True,
            "messages": state["messages"] + [{"role": "assistant", "content": delete_response}]
        }
        return updated_state

    def normal_therapy(self, state: TherapyState) -> TherapyState:
        response_message = (
            "Thank you for sharing. I hear you and I'm here to support you. Let's continue our conversation so I can help you further."
        )
        updated_state = {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": response_message}]
        }
        return updated_state

    def get_response(self, user_message: str) -> str:
        if not hasattr(self, 'state'):
            self.state = TherapyState(
                messages=[{"role": "user", "content": user_message}],
                intake_data={},
                initial_context=self.initial_context,
                intake_phase=self.intake_phase,
                intake_topics_covered=0,
                asked_topics=[],
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
        self.intake_phase = self.state["intake_phase"]
        self.intake_data = self.state["intake_data"]
        self.delete_confirmation = self.state["delete_confirmation"]
        return last_message["content"] if last_message["role"] == "assistant" else "I'm processing your message."

    def handle_crisis(self) -> str:
        return (
            "I understand you're in deep pain. If you feel unsafe, please seek immediate help or call 988 (in the US). "
            "You are not alone—I am here to support you."
        )

    def request_delete_confirmation(self, *args, **kwargs) -> str:
        self.delete_confirmation = True
        return (
            "Would you like to end this session and delete the channel? Type 'YES, END SESSION' to confirm, or continue chatting if not."
        )

    def delete_channel(self, *args, **kwargs) -> str:
        if self.channel and self.delete_confirmation:
            asyncio.create_task(self.channel.delete())
            return "Channel deletion initiated. Please take care."
        return "Deletion failed – no confirmation received."

    async def start_session(self, channel: discord.TextChannel, user: discord.Member, user_message: str):
        self.channel = channel
        logger.info(f"[INTAKE LOG] Started intake for user {user.id} with initial context: {user_message[:50]}...")
        print(f"[INTAKE LOG] Started intake for user {user.id} with initial context: {user_message[:50]}...")
        greeting_prompt = (
            f'You shared: "{user_message}"\n'
            "I truly appreciate your courage and vulnerability in sharing how you feel. "
            "Could you tell me a bit more about the emotions you're experiencing right now?"
        )
        greeting = self.llm.invoke([
            SystemMessage(content="You are a warm, compassionate therapist who listens respectfully and empathetically."),
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
            delete_confirmation=False,
            user_id=str(user.id),
            channel_id=str(channel.id),
            intent="NEUTRAL"
        )
        self.intake_phase = True
        self.initial_context = user_message
        self.intake_data = {"Presenting problem and symptoms": user_message}

    async def handle_message(self, message: discord.Message):
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