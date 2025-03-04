import os
import discord
import logging
from discord.ext import commands
from dotenv import load_dotenv

from classifier import MistralClassifier
from agent import TherapistAgent

load_dotenv()

PREFIX = "!"
logger = logging.getLogger("discord")
logging.basicConfig(level=logging.INFO)

intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

MISTRAL_API_KEY = os.getenv("MISTRAL_API")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

classifier = MistralClassifier(api_key=MISTRAL_API_KEY)

# Track active therapy sessions (channel_id -> TherapistAgent)
active_sessions = {}

@bot.event
async def on_ready():
    logger.info(f"{bot.user} has connected to Discord!")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # If the message is in an active therapy session, process it with the corresponding agent
    if message.channel.id in active_sessions:
        therapist_agent = active_sessions[message.channel.id]
        response = therapist_agent.get_response(message.content)
        
        if response.startswith(f"DELETE_CHANNEL_{message.channel.id}"):
            await message.channel.delete()
            del active_sessions[message.channel.id]
            await message.author.send("Your therapy session has ended. The channel has been deleted.")
        else:
            await message.channel.send(response)
        return

    # Process messages from other channels
    logger.info(f"Processing message from {message.author}: {message.content}")
    
    moderation_results = await classifier.moderate([message])
    
    for result in moderation_results:
        flag_message = classifier.check_all_flags(result)
        
        if flag_message:
            logger.info(f"Flag triggered for {message.author}: {flag_message}")
            
            # Check if a session already exists for this user
            user_session = next((agent for agent in active_sessions.values() if agent.user_id == message.author.id), None)
            
            if user_session:
                # Append the message to the existing session
                response = user_session.get_response(message.content)
                await message.author.send(f"Message appended to your therapy session. Agent response: {response}")
            else:
                # Create a new session
                therapy_channel = await TherapistAgent.create_private_channel(
                    guild=message.guild,
                    user=message.author,
                    bot_user=bot.user,
                )
                
                therapist_agent = TherapistAgent(channel_id=therapy_channel.id)
                await therapist_agent.start_session(therapy_channel, message.author, flag_message)
                
                # Track active session
                active_sessions[therapy_channel.id] = therapist_agent
            
            return

bot.run(DISCORD_TOKEN)
