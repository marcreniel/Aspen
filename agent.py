import logging
from mistralai import Mistral
import discord

MISTRAL_MODEL = "mistral-small-latest"
SYSTEM_PROMPT = "You are a helpful assistant."

logger = logging.getLogger("discord")

class MistralAgent:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)

    async def run(self, message: discord.Message):
        """
        Process a message with the Mistral model and return a response.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message.content},
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        return response.choices[0].message.content

    async def moderate(self, messages):
        """
        Use Mistral's moderation classifier to evaluate messages.
        """
        results = []
        for message in messages:
            response = await self.client.classifiers.moderate_async(
                model="mistral-moderation-2411",
                inputs=message.content
            )
            results.append({
                "user": message.author.name,
                "content": message.content,
                "moderation_result": response.results[0]
            })
        
        print(results)
        return results
    
    def proactive_evaluation(self, message):
        """Check based on classifier threshold on certain categories to trigger agent proactively"""
        if (
            message["moderation_result"].category_scores.get("selfharm", 0) >= 0.01 or
            message["moderation_result"].category_scores.get("hate_and_discrimination", 0) >= 0.01 or
            message["moderation_result"].category_scores.get("violence_and_threats", 0) >= 0.01
        ):
            return "Triggered Proactive Evaluation"

    def check_sexual_content(self, moderation_result):
        """Check if content is flagged for sexual content"""
        if moderation_result["moderation_result"].categories.get("sexual", False):
            return "Triggered (sexual content)"
        return None
    
    def check_hate_discrimination(self, moderation_result):
        """Check if content is flagged for hate and discrimination"""
        if moderation_result["moderation_result"].categories.get("hate_and_discrimination", False):
            return "Triggered (hate and discrimination)"
        return None
    
    def check_violence_threats(self, moderation_result):
        """Check if content is flagged for violence and threats"""
        if moderation_result["moderation_result"].categories.get("violence_and_threats", False):
            return "Triggered (violence and threats)"
        return None
    
    def check_dangerous_criminal(self, moderation_result):
        """Check if content is flagged for dangerous and criminal content"""
        if moderation_result["moderation_result"].categories.get("dangerous_and_criminal_content", False):
            return "Triggered (dangerous and criminal content)"
        return None
    
    def check_selfharm(self, moderation_result):
        """Check if content is flagged for self-harm"""
        if moderation_result["moderation_result"].categories.get("selfharm", False):
            return "Triggered (self-harm)"
        return None
    
    def check_health(self, moderation_result):
        """Check if content is flagged for health misinformation"""
        if moderation_result["moderation_result"].categories.get("health", False):
            return "Triggered (health misinformation)"
        return None
    
    def check_pii(self, moderation_result):
        """Check if content is flagged for personal identifiable information"""
        if moderation_result["moderation_result"].categories.get("pii", False):
            return "Triggered (personal identifiable information)"
        return None
    
    def check_all_flags(self, moderation_result):
        """Check all moderation flags and return appropriate message if any are triggered"""
        checks = [
            self.check_selfharm,
            self.check_hate_discrimination,
            self.check_violence_threats,
            self.check_sexual_content,
            self.check_dangerous_criminal,
            self.check_health,
            self.check_pii,
            self.proactive_evaluation,
        ]
        
        for check in checks:
            result = check(moderation_result)
            if result:
                return result
        
        return None
