from openai import OpenAI
client = OpenAI()

def ai_answer(query, df, chiller):
    recent = df[df["chiller"] == chiller].tail(12)

    prompt = f"""
    Query: {query}
    Chiller: {chiller}

    Last 12 readings:
    {recent.to_string()}

    Provide:
    1. Query answer
    2. Reason for deviation in graph
    3. Identify which parameter (ambient, IT load, CHW inlet) caused deviation
    4. Operational impact
    5. Clear recommended action
    """

    try:
        out = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return out.choices[0].message.content

    except Exception as e:
        return f"AI Error: {e}"
