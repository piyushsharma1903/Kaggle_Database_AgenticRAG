import json
import os
import re 
from openai import OpenAI 
from tools import (
    semantic_retrieve,
    get_avg_stay,
    count_patients,
    compare_stay
)

def safe_json_extract(text: str):
    if not text:
        return {}

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except:
        pass

    import re
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return {}



client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com") #set your DeepSeek API key here #function_json ke liye ye krna he hoga

# ---- Tool Schema Exposed to LLM ----
FUNCTIONS = [
    {
        "name": "semantic_retrieve",
        "description": "Retrieve semantically relevant summaries using vector search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "topk": {"type": "number"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_avg_stay",
        "description": "Compute average hospital stay for a specific medical condition",
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {"type": "string"}
            },
            "required": ["condition"]
        }
    },
    {
        "name": "count_patients",
        "description": "Count patients using structured filters like medical condition or stay_length",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_obj": {"type": "object"}
            },
            "required": ["filter_obj"]
        }
    },
    {
        "name": "compare_stay",
        "description": "Compare average stay lengths between two medical conditions",
        "parameters": {
            "type": "object",
            "properties": {
                "cond1": {"type": "string"},
                "cond2": {"type": "string"}
            },
            "required": ["cond1", "cond2"]
        }
    }
]

def llm_router(user_query: str):
    system_prompt = """
You are a Query Router for a Medical RAG system.

YOU MUST FOLLOW THESE RULES STRICTLY:

1. Reply ONLY in clean JSON.
2. NO markdown. NO backticks. NO explanations. NO comments.
3. Only output this structure:

{
  "function": "<tool_name>",
  "args": { ... }
}

VALID TOOLS & ARGUMENTS:

1. semantic_retrieve
   args: { "query": "<text>", "topk": 5 }

2. get_avg_stay
   args: { "condition": "<condition>" }

3. count_patients
   args: {
     "filter_obj": {
         "Medical Condition": "<condition>",
         "stay_length": { "$gt" or "$lt" or "$gte" or "$lte": <number> }
     }
   }

4. compare_stay
   args: { "cond1": "<condition>", "cond2": "<condition>" }

ROUTING RULES:
- If question contains "how many", "count", "number" → count_patients
- If question asks for average stay → get_avg_stay
- If question compares two conditions → compare_stay
- If question is fuzzy/descriptive → semantic_retrieve
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )

    return response.choices[0].message


#executor
def execute_function(message):
    # Parse JSON from router
    data = safe_json_extract(message.content)
    if not data:
        return None

    # Accept variants like function_name instead of function
    name = data.get("function") or data.get("function_name")
    args = data.get("args") or data.get("parameters") or {}

    if not name:
        return None

    # --- STANDARDIZE count_patients arguments ---
    if name == "count_patients":
        filter_obj = args.get("filter_obj", {})
        # if router gave "condition" and "min_stay_days", convert it
        if "condition" in args or "min_stay_days" in args:
            cond = args.get("condition")
            days = args.get("min_stay_days")

            
            if cond:
                filter_obj["Medical Condition"] = cond.capitalize()

            if days is not None:
                filter_obj["stay_length"] = {"$gt": int(days)}

        if "Medical Condition" in filter_obj:
            mc = filter_obj["Medical Condition"]
            if isinstance(mc, str):
                filter_obj["Medical Condition"] = mc.capitalize()
        
        args={"filter_obj": filter_obj}

        return count_patients(**args)

    # --- SEMANTIC RETRIEVAL ---
    if name == "semantic_retrieve":
        if "topk" not in args:
            args["topk"] = 5
        return semantic_retrieve(**args)

    # --- AVERAGE STAY ---
    if name == "get_avg_stay":
        return get_avg_stay(**args)

    # --- COMPARISON ---
    if name == "compare_stay":
        return compare_stay(**args)

    return None



def llm_explainer(user_query, tool_result):
    result_text = json.dumps(tool_result, ensure_ascii=False)

    messages = [
        {"role": "system", "content": "You explain the tool result only. No hallucination."},
        {"role": "user", "content": user_query},
        {"role": "system", "content": f"tool_result: {result_text}"}
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=1.0
    )
    return response.choices[0].message.content


#user input
if __name__ == "__main__":
    while True:
        query = input("\nAsk something: ")

        # 1. Route the query → which tool?
        plan = llm_router(query)
        print("\n[Router Decision]:", plan)

        # 2. Execute the selected tool
        tool_output = execute_function(plan)
        print("\n[Tool Output]:", tool_output)

        # 3. LLM explains the result to user
        final_answer = llm_explainer(query, tool_output)

        print("\n=== FINAL ANSWER ===\n")
        print(final_answer)
