import discord
from discord.ext import commands, voice_recv
import os
from dotenv import load_dotenv
import logging
import asyncio

logger = logging.getLogger("discord")
logger.setLevel(logging.INFO)

discord.opus.load_opus('/opt/homebrew/Cellar/opus/1.5.2/lib/libopus.dylib')
if not discord.opus.is_loaded():
    raise RuntimeError('Opus failed to load')

load_dotenv()

token = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.all()
intents.members = True

bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents)

class MySink(voice_recv.AudioSink):
    def __init__(self, bot):
        self.bot = bot
        super().__init__()

    def wants_opus(self):
        return False

    def write(self, user, data):
        pass

    def cleanup(self):
        pass

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member: discord.Member):
        logger.info(f"{member.name} started speaking in {member.voice.channel.name}")

class Testing(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def test(self, ctx):
        vc = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        vc.listen(MySink(self.bot))

    @commands.command()
    async def stop(self, ctx):
        await ctx.voice_client.disconnect()

    @commands.command()
    async def die(self, ctx):
        if ctx.voice_client:
            ctx.voice_client.stop()
        await ctx.bot.close()

@bot.event
async def on_ready():
    logger.info('Logged in as {0.id}/{0}'.format(bot.user))
    print('------')

async def setup_hook():
    await bot.add_cog(Testing(bot))

@bot.event
async def on_connect():
    await setup_hook()

bot.run(token)
