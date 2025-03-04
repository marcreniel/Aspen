import os
import discord
import logging
from discord.ext import commands
from dotenv import load_dotenv

from agent import MistralAgent

PREFIX = "!"

logger = logging.getLogger("discord")
logging.basicConfig(level=logging.INFO)

load_dotenv()

intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

MISTRAL_API_KEY = os.getenv("MISTRAL_API")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

agent = MistralAgent(api_key=MISTRAL_API_KEY)

@bot.event
async def on_ready():
    """
    Called when the bot successfully connects to Discord.
    """
    logger.info(f"{bot.user} has connected to Discord!")

@bot.event
async def on_message(message: discord.Message):
    """
    Handle incoming messages.
    """
    await bot.process_commands(message)

    if message.author.bot or message.content.startswith(PREFIX):
        return

    logger.info(f"Processing message from {message.author}: {message.content}")
    
    moderation_results = await agent.moderate([message])
    
    for result in moderation_results:
        flag_message = agent.check_all_flags(result)
        if flag_message:
            await message.reply(flag_message)
            return 
    
bot.run(DISCORD_TOKEN)
