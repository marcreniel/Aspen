import os
import discord
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")

class TherapistAgent:
    def __init__(self):
        # Initialize the LLM with OpenAI API
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
        
        # Define the system prompt
        self.system_prompt = SystemMessage(content=(
            "You are a compassionate therapist dedicated to providing empathetic emotional support. "
            "Listen carefully, ask clarifying questions if needed, and help users explore their feelings in a non-judgmental manner."
        ))
        
        # Create a chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
        # Create the conversation chain
        self.chain = (
            RunnablePassthrough.assign(chat_history=lambda x: x.get("chat_history", []))
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.chat_history = []

    def get_response(self, user_message: str) -> str:
        """
        Generate a response from the therapist agent for a given user message.
        """
        response = self.chain.invoke({"input": user_message, "chat_history": self.chat_history})
        self.chat_history.extend([HumanMessage(content=user_message), AIMessage(content=response)])
        return response

    @staticmethod
    async def create_private_channel(guild: discord.Guild, user: discord.Member, bot_user: discord.ClientUser):
        """
        Create a private text channel for a therapy session.
        """
        # Define permission overwrites so only the user and bot can access the channel
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(view_channel=False),
            user: discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True),
            bot_user: discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True),
        }
        
        # Create a private channel named after the user
        channel_name = f"therapy-{user.name}".replace(" ", "-")
        therapy_channel = await guild.create_text_channel(name=channel_name, overwrites=overwrites)
        
        return therapy_channel

    @staticmethod
    async def delete_private_channel(channel: discord.TextChannel):
        """
        Delete a private therapy session channel.
        """
        await channel.delete()
