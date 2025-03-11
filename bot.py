import os
import discord
import logging
import asyncio
import tempfile
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

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

classifier = MistralClassifier(api_key=MISTRAL_API_KEY)

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
        
        # Check for deletion confirmation command
        if therapist_agent.delete_confirmation and message.content.strip().upper() == "YES, END SESSION":
            # Create a temporary file to write the conversation summary and log to.
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
                conversation_history = ""
                for msg in therapist_agent.state["messages"]:
                    conversation_history += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

                conversation_summary = therapist_agent.get_conversation_summary(conversation_history)
                
                temp_file.write(f"SUMMARY:\n\n{conversation_summary}\n\nCHAT LOG:\n\n{conversation_history}")
                temp_file_name = temp_file.name

            # Get the user by their ID.
            user = message.guild.get_member(int(therapist_agent.user_id))
            if user:
                # Send the file to the user via DM before deleting the channel.
                await user.send("Here is a summary and log of your conversation:", file=discord.File(temp_file_name))

            await message.channel.send("Thank you for confirming. This therapy channel will now be deleted. Take care!")
            await asyncio.sleep(5)  # Allow the user some time to read the message
            await message.channel.delete()
            del active_sessions[message.channel.id]
            return

        response = therapist_agent.get_response(message.content)

        # Send response according to Discord's character limit
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
                    # Append the message to the existing therapy session
                    response = user_session.get_response(message.content)
                    therapy_channel = bot.get_channel(user_session.channel_id)
                    
                    if therapy_channel:
                        await therapy_channel.send(f"**{message.author.name}:** {message.content}")
                        await therapy_channel.send(f"**Therapist:** {response}")
                    
                    await message.author.send("Your message has been added to your session.")
                    return
                else:
                    # Create a new private therapy channel and session
                    therapy_channel = await TherapistAgent.create_private_channel(
                        guild=message.guild,
                        user=message.author,
                        bot_user=bot.user,
                    )
                    therapist_agent = TherapistAgent(channel_id=therapy_channel.id, user_id=message.author.id)
                    # Pass the actual message content instead of just the flag type
                    await therapist_agent.start_session(therapy_channel, message.author, message.content)
                    active_sessions[therapy_channel.id] = therapist_agent
            else:
                # For harmful language not covered by crisis or deletion,
                # generate and send a compassionate GPT warning.
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
        
@bot.event
async def on_guild_channel_delete(channel: discord.abc.GuildChannel):
    if channel.id in active_sessions:
        del active_sessions[channel.id]
        logger.info(f"Active session for channel {channel.id} has been removed after channel deletion.")

bot.run(DISCORD_TOKEN)
