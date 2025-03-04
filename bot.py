import os
import discord
import logging
import asyncio
from dotenv import load_dotenv
from discord.ext import commands
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

    # Check if the message is in an active therapy session channel
    if message.channel.id in active_sessions:
        therapist_agent = active_sessions[message.channel.id]
        
        if therapist_agent.delete_confirmation and message.content.strip().upper() == "YES, END SESSION":
            await message.channel.send("Thank you for confirming. This therapy channel will now be deleted. Take care!")
            await asyncio.sleep(5)  # Give the user a moment to read the message
            await message.channel.delete()
            del active_sessions[message.channel.id]
            return

        response = therapist_agent.get_response(message.content)
        await message.channel.send(response)
        return
    # If not in an active session channel, check if the user has an active session
    user_session = next((agent for agent in active_sessions.values() if agent.user_id == message.author.id), None)
    
    # If no active session, proceed with moderation and potential new session creation
    logger.info(f"Processing message from {message.author}: {message.content}")
    
    moderation_results = await classifier.moderate([message])
    
    for result in moderation_results:
        flag_message = classifier.check_all_flags(result)
        
        if flag_message:
            logger.info(f"Flag triggered for {message.author}: {flag_message}")
            
            if flag_message == "ProactiveEvaluation" or flag_message == "SelfHarm":
                if user_session:
                    # Append the message to the existing therapy session
                    response = user_session.get_response(message.content)
                    therapy_channel = bot.get_channel(user_session.channel_id)
                    
                    if therapy_channel:
                        await therapy_channel.send(f"**{message.author.name}:** {message.content}")
                        await therapy_channel.send(f"**Therapist:** {response}")
                    
                    await message.author.send(f"Placeholder for message")
                    return
                else:
                    therapy_channel = await TherapistAgent.create_private_channel(
                        guild=message.guild,
                        user=message.author,
                        bot_user=bot.user,
                    )
                    
                    therapist_agent = TherapistAgent(channel_id=therapy_channel.id, user_id=message.author.id)
                    await therapist_agent.start_session(therapy_channel, message.author, flag_message)
                    
                    active_sessions[therapy_channel.id] = therapist_agent
            else:
                await message.author.send(f"Harmful languagae detected.")
                return
            return

@bot.event
async def on_guild_channel_delete(channel: discord.abc.GuildChannel):
    if channel.id in active_sessions:
        del active_sessions[channel.id]
        logger.info(f"Active session for channel {channel.id} has been removed following channel deletion.")

bot.run(DISCORD_TOKEN)
