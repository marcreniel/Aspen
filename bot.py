import os
import discord
import logging
from discord.ext import commands
from dotenv import load_dotenv

from classifier import MistralClassifier  # Existing classifier remains unchanged
from agent import TherapistAgent  # Import new agent functionality

load_dotenv()

PREFIX = "!"
logger = logging.getLogger("discord")
logging.basicConfig(level=logging.INFO)

intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

MISTRAL_API_KEY = os.getenv("MISTRAL_API")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

classifier = MistralClassifier(api_key=MISTRAL_API_KEY)

# Track active therapy sessions (channel_id -> user_id)
active_sessions = {}

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
    
    # Use classifier to moderate messages and check for flags
    moderation_results = await classifier.moderate([message])
    
    for result in moderation_results:
        flag_message = classifier.check_all_flags(result)
        
        if flag_message:
            logger.info(f"Flag triggered for {message.author}: {flag_message}")
            
            # Check if a session already exists for this user
            if any(user_id == message.author.id for user_id in active_sessions.values()):
                await message.reply("You already have an active therapy session.")
                return
            
            # Create a private therapy session channel using TherapistAgent's helper method
            therapy_channel = await TherapistAgent.create_private_channel(
                guild=message.guild,
                user=message.author,
                bot_user=bot.user,
            )
            
            # Track active session (channel_id -> user_id)
            active_sessions[therapy_channel.id] = message.author.id
            
            # Send initial messages in the new channel
            await therapy_channel.send(
                f"Hello {message.author.mention}, welcome to your private therapy session. "
                "This session is powered by our compassionate AI therapist. "
                "Feel free to share what's on your mind."
            )
            
            await message.reply(f"A private therapy channel has been created for you: {therapy_channel.mention}")
            
            return
    
    # Route messages in active therapy channels through the therapist agent
    if message.channel.id in active_sessions:
        therapist_agent = TherapistAgent()  # Create an instance of the therapist agent
        
        response = therapist_agent.get_response(message.content)
        
        await message.channel.send(response)

@bot.command(name="delete_channel")
async def delete_channel(ctx):
    """
    Deletes a therapy session channel if it corresponds to the requesting user or an admin.
    """
    if ctx.channel.id in active_sessions:
        if active_sessions[ctx.channel.id] == ctx.author.id or ctx.author.guild_permissions.administrator:
            await TherapistAgent.delete_private_channel(ctx.channel)
            active_sessions.pop(ctx.channel.id, None)
            logger.info(f"Deleted therapy session channel: {ctx.channel.name}")
        else:
            await ctx.send("You do not have permission to delete this channel.", delete_after=5)
    else:
        await ctx.send("This is not a therapy session channel.", delete_after=5)

bot.run(DISCORD_TOKEN)
