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

        self.system_prompt = SystemMessage(content=(
            "You are a compassionate therapist dedicated to providing empathetic emotional support. "
            "Listen carefully, ask clarifying questions if needed, and help users explore their feelings in a non-judgmental manner. "
            "The first message you receive will be a moderation flag. Use this information to guide your initial response, "
            "but do not explicitly mention the flag to the user. "
            "If the user expresses a clear intent to end the session, use the delete_channel tool."
        ))
        
        self.prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
        self.chat_history = []

        self.delete_channel_tool = Tool(
            name="delete_channel",
            func=self.delete_channel,
            description="Delete the therapy channel when the user wants to end the session"
        )

        self.agent = initialize_agent(
            [self.delete_channel_tool],
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True
        )

    def get_response(self, user_message: str) -> str:
        response = self.agent.run(input=user_message, chat_history=self.chat_history)
        self.chat_history.extend([HumanMessage(content=user_message), AIMessage(content=response)])
        return response

    def delete_channel(self, *args, **kwargs) -> str:
        if self.channel is not None:
            asyncio.create_task(self.channel.delete())
            return f"The therapy channel has been deleted."
        return "No channel is set to delete."

    @staticmethod
    async def create_private_channel(guild: discord.Guild, user: discord.Member, bot_user: discord.ClientUser) -> discord.TextChannel:
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(view_channel=False),
            user: discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True),
            bot_user: discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True),
        }
        
        channel_name = f"therapy-{user.name}".replace(" ", "-")
        therapy_channel = await guild.create_text_channel(name=channel_name, overwrites=overwrites)
        return therapy_channel

    async def start_session(self, channel: discord.TextChannel, user: discord.Member, moderation_flag: str):
        self.channel = channel
        await user.send(f"A private therapy channel has been created for you: {channel.mention}")
        initial_response = self.get_response(moderation_flag)
        await channel.send(initial_response)
