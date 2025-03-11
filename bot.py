import os
import discord
import logging
import asyncio
import tempfile
import json
import atexit
from dotenv import load_dotenv
from discord.ext import commands, tasks
from classifier import MistralClassifier
from agent import TherapistAgent

load_dotenv()

PREFIX = "!"
logger = logging.getLogger("discord")
logging.basicConfig(level=logging.INFO)

intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

classifier = MistralClassifier(api_key=MISTRAL_API_KEY)

# Global dictionary to store active sessions
active_sessions = {}

# File to store session data persistently
ACTIVE_SESSIONS_FILE = 'active_sessions.json'

# Helper functions to serialize and deserialize session data
def save_active_sessions():
    serialized_sessions = {}
    for channel_id, therapist_agent in active_sessions.items():
        # It is assumed that TherapistAgent has a method to extract its serializable data.
        serialized_sessions[channel_id] = therapist_agent.get_serializable_data()
    with open(ACTIVE_SESSIONS_FILE, 'w') as f:
        json.dump(serialized_sessions, f, indent=4)
    logger.info("Active sessions saved to disk.")

def load_active_sessions():
    if os.path.exists(ACTIVE_SESSIONS_FILE):
        with open(ACTIVE_SESSIONS_FILE, 'r') as f:
            data = json.load(f)
        for channel_id_str, session_data in data.items():
            channel_id = int(channel_id_str)
            # It is assumed that TherapistAgent has a class method to create an instance from saved data.
            therapist_agent = TherapistAgent.from_serializable_data(session_data)
            active_sessions[channel_id] = therapist_agent
        logger.info("Active sessions loaded from disk.")

# Register to save sessions on exit
atexit.register(save_active_sessions)

# Optional: Auto-save sessions periodically (e.g., every 60 seconds)
@tasks.loop(seconds=60)
async def autosave_active_sessions():
    save_active_sessions()

@bot.event
async def on_ready():
    logger.info(f"{bot.user} has connected to Discord!")
    load_active_sessions()  # Load sessions when the bot starts
    autosave_active_sessions.start()  # Start periodic autosave

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Check if the message is in an active therapy session channel
    if message.channel.id in active_sessions:
        therapist_agent = active_sessions[message.channel.id]

        # Check for deletion confirmation command
        if therapist_agent.delete_confirmation and message.content.strip().upper() == "YES, END SESSION":
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
                conversation_history = ""
                for msg in therapist_agent.state["messages"]:
                    conversation_history += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

                conversation_summary = therapist_agent.get_conversation_summary(conversation_history)
                temp_file.write(f"SUMMARY:\n\n{conversation_summary}\n\nCHAT LOG:\n\n{conversation_history}")
                temp_file_name = temp_file.name

            user = message.guild.get_member(int(therapist_agent.user_id))
            if user:
                await user.send(
                    "We hope you feel better after our conversation. Here is a summary and log of your conversation, for your convenience. Take care.",
                    file=discord.File(temp_file_name)
                )

            await message.channel.send("Thank you for confirming. This therapy channel will now be deleted. Take care!")
            await asyncio.sleep(5)
            await message.channel.delete()
            del active_sessions[message.channel.id]
            return

        response = therapist_agent.get_response(message.content)
        if len(response) <= 2000:
            await message.channel.send(response)
        else:
            split_message = [response[i:i+2000] for i in range(0, len(response), 2000)]
            for split in split_message:
                await message.channel.send(split)
                await asyncio.sleep(2)
        return

    # If not in an active session channel, check if the user already has an active session
    user_session = next((agent for agent in active_sessions.values() if agent.user_id == message.author.id), None)

    logger.info(f"Processing message from {message.author}: {message.content}")
    moderation_results = await classifier.moderate([message])
    
    for result in moderation_results:
        flag_message = classifier.check_all_flags(result)
        if flag_message:
            logger.info(f"Flag triggered for {message.author}: {flag_message}")
            if flag_message == "ProactiveEvaluation" or flag_message == "SelfHarm":
                if user_session:
                    response = user_session.get_response(message.content)
                    therapy_channel = bot.get_channel(user_session.channel_id)
                    if therapy_channel:
                        await therapy_channel.send(f"**{message.author.name}:** {message.content}")
                        await therapy_channel.send(f"**Therapist:** {response}")
                    await message.author.send("Your message has been added to your session.")
                    return
                else:
                    therapy_channel = await TherapistAgent.create_private_channel(
                        guild=message.guild,
                        user=message.author,
                        bot_user=bot.user,
                    )
                    therapist_agent = TherapistAgent(channel_id=therapy_channel.id, user_id=message.author.id)
                    await therapist_agent.start_session(therapy_channel, message.author, message.content)
                    active_sessions[therapy_channel.id] = therapist_agent
            else:
                warning_message = TherapistAgent.generate_warning_response(message.content)
                await message.author.send(warning_message)
                try:
                    await message.delete()
                    logger.info(f"Deleted flagged message from {message.author}")
                except discord.errors.Forbidden:
                    logger.warning(f"Bot doesn't have permission to delete message from {message.author}")
                except discord.errors.NotFound:
                    logger.warning(f"Message from {message.author} not found (already deleted?)")
                return
            return

    await bot.process_commands(message)

@bot.event
async def on_guild_channel_delete(channel: discord.abc.GuildChannel):
    if channel.id in active_sessions:
        del active_sessions[channel.id]
        logger.info(f"Active session for channel {channel.id} has been removed after channel deletion.")

@bot.command(name="tutorial")
async def therapy_help_command(ctx):
    embed = discord.Embed(
        title="Aspen Bot Help",
        description="Instructions on how to trigger the therapy response.",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="Triggering a Therapy Session",
        value=(
            "Simply send a message in any channel. Your message will be analyzed, "
            "and if it is flagged for concerns, "
            "a private therapy session will be automatically initiated."
        ),
        inline=False
    )
    embed.add_field(
        name="Active Therapy Session",
        value=(
            "If you already have an active therapy session, any further messages will be "
            "appended to it so that you receive ongoing support."
        ),
        inline=False
    )
    embed.add_field(
        name="Ending a Session",
        value=(
            "To end your therapy session, type **YES, END SESSION** in your therapy channel. "
            "A summary and full log of the conversation will be sent to your DM before the channel is deleted."
        ),
        inline=False
    )
    await ctx.send(embed=embed)

bot.run(DISCORD_TOKEN)
