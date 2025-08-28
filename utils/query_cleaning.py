import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
class QueryCleaner:
    def __init__(self, client=None):
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = client


    def cleaning_kis(self, query:str):
        prompt = f"""
        Rewrite the following query in English, optimized for CLIP video retrieval. 

        Guidelines:
        - Keep it short and descriptive.
        - Focus on the main visual content.
        - Do not add explanations or extra words.
        - Return only the cleaned query, no quotes.

        Original query: "{query}"
        """
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",  # hoặc model nhỏ hơn
            messages=[{"role": "user", "content": prompt}],
        )
        cleaned = resp.choices[0].message.content.strip()
        
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        
        return cleaned


    def cleaning_qa(self, query: str):
        prompt = f"""
        You are an assistant that rewrites queries for video retrieval and QA.

        ## Instructions
        1. You are given a query in any language.
        2. First, translate the query into English if it is not in English.
        3. Create two outputs:
        - Step 0: Rewrite the query into a short, descriptive, and visual-oriented sentence optimized for CLIP retrieval (for KIS).
        - Step 1: Provide the full translated query in clear English (for Answering).

        ## Constraints
        - Output strictly in English only.
        - Do not add explanations, commentary, or extra text.
        - Ensure output follows the required numbering format.
        - Each step must be in one line only.

        ## Output Format
        Return the result strictly as a numbered list like this:

        0. <rewritten cleaned query for KIS>
        1. <full translated query for Answering>

        ## Input Query
        {query}
        """

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()

        subqueries = []
        for line in text.splitlines():
            if line.strip() and (line[0].isdigit() or line.startswith("-")):
                subqueries.append(line.split(".", 1)[-1].strip())
            else:
                subqueries.append(line.strip())

        return [q for q in subqueries if q]


    def cleaning_trake(self, query: str):
        prompt = f"""
        You are an assistant that rewrites and structures queries for video retrieval.

        ## Instructions
        1. You are given a query in any language.
        2. First, translate the query into English. 
        - This full translation is used internally to extract the detailed steps.
        3. Next, rewrite the translated query into a short, descriptive, and visual-oriented sentence optimized for CLIP retrieval. 
        - This rewrite will be Step 0.
        4. Finally, decompose the full translation into multiple subqueries, each describing one step of the action.
        - Each subquery must be short, visual-oriented, and descriptive of a single action or moment.
        - These subqueries will be Step 1, Step 2, ..., Step N.

        ## Constraints
        - Output strictly in English only.
        - Do not add explanations, commentary, or extra text.
        - Step 0 must come from the rewritten version, not the raw translation.
        - Steps 1..N must come from the full translation.
        - Each step must be in one line only.

        ## Output Format
        Return the result strictly as a numbered list like this:

        0. <rewritten cleaned query>
        1. <subquery for step 1>
        2. <subquery for step 2>
        ...
        N. <subquery for step N>

        ## Input Query
        {query}
        """
                
        resp = self.client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()

        subqueries = []
        for line in text.splitlines():
            if line.strip() and (line[0].isdigit() or line.startswith("-")):
                subqueries.append(line.split(".", 1)[-1].strip())
            else:
                subqueries.append(line.strip())
        return [q for q in subqueries if q]