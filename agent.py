import os
import discord
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")

class TherapistAgent:
    def __init__(self, channel_id, user_id=None):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
        self.channel_id = channel_id
        self.user_id = user_id
        self.channel = None
        self.delete_confirmation = False

        # Enhanced system prompt for therapy flow
        self.system_prompt = SystemMessage(content=(
            "You are a compassionate therapist prioritizing user safety. "
            "First detect emotional crises, then handle technical requests. "
            "Never mention message analysis to the user. Maintain natural flow."
        ))

        self.prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        self.chat_history = []

        # Tools setup for deletion confirmation and channel deletion
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

        # Attributes for natural intake conversation
        self.intake_phase = False
        self.intake_fields = []   # Topics to cover during intake
        self.intake_prompts = []  # Natural, conversational prompts for each topic
        self.intake_data = {}     # To store user responses for each field
        self.current_intake_index = 0
        self.initial_context = ""  # User’s initial message (presenting problem)

    def analyze_intent(self, message: str) -> str:
        intent_prompt = """Analyze message for these intents:
        1. CRISIS - Expressions of self-harm/suicide/extreme hopelessness
        2. DELETE_REQUEST - Clear channel management requests
        3. NEUTRAL - Other messages

        Examples:
        - "delete this" → DELETE_REQUEST
        - "I want to end it all" → CRISIS
        - "Goodbye" → NEUTRAL

        Respond ONLY with: CRISIS, DELETE_REQUEST, or NEUTRAL"""
        
        crisis_check = self.llm.invoke([
            SystemMessage(content=intent_prompt),
            HumanMessage(content=f"Message: {message}")
        ])
        if "CRISIS" in crisis_check.content:
            return "CRISIS"
        
        deletion_check = self.llm.invoke([
            SystemMessage(content="Is this a request to delete/close the channel? Respond ONLY with YES or NO"),
            HumanMessage(content=message)
        ])
        return "DELETE_REQUEST" if "YES" in deletion_check.content.upper() else "NEUTRAL"

    def get_response(self, user_message: str) -> str:
        # If we are in the midst of collecting intake details, use a natural conversation flow.
        if self.intake_phase:
            if self.current_intake_index < len(self.intake_fields):
                current_field = self.intake_fields[self.current_intake_index]
                # Record user response or assign a default if they opt to skip.
                if user_message.strip().lower() in ["skip", "n/a", "no comment"]:
                    self.intake_data[current_field] = "Not provided by user"
                else:
                    self.intake_data[current_field] = user_message
                self.current_intake_index += 1

                # Ask the next question in a conversational tone.
                if self.current_intake_index < len(self.intake_fields):
                    next_prompt = self.intake_prompts[self.current_intake_index]
                    return next_prompt
                else:
                    self.intake_phase = False
                    summary = "I really appreciate you opening up. Here's a quick summary of what you've shared:\n"
                    summary += f"• Presenting problem and symptoms: {self.initial_context}\n"
                    for field in self.intake_fields:
                        response = self.intake_data.get(field, "Not provided")
                        summary += f"• {field}: {response}\n"
                    summary += "\nThank you for trusting me with this information. We can continue our conversation now."
                    return summary
            else:
                self.intake_phase = False
                return "It seems we've covered everything. How are you feeling now?"
        else:
            # Normal conversation outside the intake phase.
            intent = self.analyze_intent(user_message)
            if intent == "CRISIS":
                return self.handle_crisis()
            if intent == "DELETE_REQUEST":
                return self.request_delete_confirmation()
            
            response = self.agent.run(
                input=user_message,
                chat_history=self.chat_history
            )
            self.chat_history.extend([
                HumanMessage(content=user_message),
                AIMessage(content=response)
            ])
            return response

    def handle_crisis(self) -> str:
        return (
            "I hear you're in significant pain. Please know you're not alone. "
            "For immediate help, contact:\n"
            "-  988 Suicide & Crisis Lifeline (US)\n"
            "-  Crisis Text Line: TEXT 'HOME' to 741741\n"
            "-  International help: https://findahelpline.com\n\n"
            "I'm here to listen if you want to talk more."
        )

    def request_delete_confirmation(self, *args, **kwargs) -> str:
        self.delete_confirmation = True
        return (
            "Would you like to end this therapy session and delete the channel? "
            "Type `YES, END SESSION` to confirm, or continue chatting if not."
        )

    def delete_channel(self, *args, **kwargs) -> str:
        if self.channel and self.delete_confirmation:
            asyncio.create_task(self.channel.delete())
            return "Channel deletion initiated. Take care of yourself."
        return "Deletion failed - no confirmation received."

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

    async def start_session(self, channel: discord.TextChannel, user: discord.Member, flagged_message: str):
        self.channel = channel
        # If the flagged message indicates self-harm or a proactive evaluation,
        # begin a natural intake conversation.
        if flagged_message in ["SelfHarm", "ProactiveEvaluation"]:
            self.intake_phase = True
            # Here we assume flagged_message brings the user’s own words, e.g., "I'm feeling really depressed today"
            self.initial_context = flagged_message
            self.intake_data["Presenting problem and symptoms"] = self.initial_context
            
            # Define the topics and corresponding natural prompts.
            self.intake_fields = [
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
            self.intake_prompts = [
                "To help me understand you better, could you share a bit about your personal or family history with mental health? There's no pressure if you'd rather not say.",
                "Thanks for sharing that. How have things been in your daily life—maybe with work, relationships, or any recent stresses?",
                "I appreciate you opening up. Have you ever tried therapy or other ways to cope during tough times?",
                "Sometimes, people may use substances to ease emotional pain. If you're comfortable, could you share if that has been part of your experience?",
                "Your well-being matters a lot. When things get overwhelming, have you ever experienced thoughts of self-harm or felt extremely low?",
                "Looking ahead, what changes or improvements would you hope to see in your life? Even small hopes can be important.",
                "Our cultural background can affect how we experience challenges. Would you like to share how your cultural or personal beliefs influence your outlook?",
                "Sometimes medications or health conditions influence our mood. Are you currently taking any medications or dealing with health issues you'd like to mention?",
                "Lastly, it can be really helpful to know who supports you during difficult times. Who do you usually rely on—friends, family, or other community support?"
            ]
            self.current_intake_index = 0
            greeting_message = (
                f"Hi {user.name}, I'm really sorry you're feeling this way. You mentioned: \"{self.initial_context}\". "
                "I want to understand you better so I can help. Let's start by talking about your personal or family history with mental health. " +
                self.intake_prompts[0]
            )
            await channel.send(greeting_message)
        else:
            await user.send(f"Private session started: {channel.mention}")
            initial_response = self.get_response(flagged_message)
            await channel.send(initial_response)

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
        system_prompt = SystemMessage(
            content="You are a thoughtful moderator. Generate a clear, compassionate, and instructive warning message that explains why using harmful language is unacceptable, the negative effects it may have on both the speaker and the community, and encourage a positive change. Also, state explicitly that this is a warning."
        )
        human_message = HumanMessage(content=f"User said: {user_message}")
        response = llm.invoke([system_prompt, human_message])
        return response.content
