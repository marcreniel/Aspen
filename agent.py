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
        
        # First check for crisis language
        crisis_check = self.llm.invoke([
            SystemMessage(content=intent_prompt),
            HumanMessage(content=f"Message: {message}")
        ])
        if "CRISIS" in crisis_check.content:
            return "CRISIS"
        
        # Then check for deletion requests
        deletion_check = self.llm.invoke([
            SystemMessage(content="Is this a request to delete/close the channel? Respond ONLY with YES or NO"),
            HumanMessage(content=message)
        ])
        return "DELETE_REQUEST" if "YES" in deletion_check.content.upper() else "NEUTRAL"

    def get_response(self, user_message: str) -> str:
        intent = self.analyze_intent(user_message)
        if intent == "CRISIS":
            return self.handle_crisis()
        if intent == "DELETE_REQUEST":
            return self.request_delete_confirmation()
        
        # Normal therapy conversation
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

    async def start_session(self, channel: discord.TextChannel, user: discord.Member, moderation_flag: str):
        self.channel = channel
        await user.send(f"Private session started: {channel.mention}")
        initial_response = self.get_response(moderation_flag)
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
        """
        Generates a compassionate warning message using GPT to explain why
        using harmful language is unacceptable.
        """
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
        system_prompt = SystemMessage(
            content="You are a thoughtful moderator. Generate a clear, compassionate, and instructive warning message that explains why using harmful language is unacceptable, the negative effects it may have on both the speaker and the community, and encourage a positive change. Also, state explicitly that this is a warning."
        )
        human_message = HumanMessage(content=f"User said: {user_message}")
        response = llm.invoke([system_prompt, human_message])
        return response.content
