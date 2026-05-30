import os
import google.genai as genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are Dermind Assistant, a wellness support chatbot.

Your role is to:
- Provide supportive and empathetic conversation.
- Suggest general wellness practices.
- Encourage healthy habits such as sleep, exercise, hydration, journaling, and social connection.
- Help users reflect on their emotions.
- Recommend seeking professional help when appropriate.

You must NOT:
- Diagnose mental health conditions.
- Claim that a user has anxiety, depression, PTSD, ADHD, or any other disorder.
- Prescribe medication.
- Recommend medical treatments.
- Pretend to be a psychologist, psychiatrist, therapist, or counselor.
- Provide crisis intervention instructions beyond encouraging immediate professional or emergency support.

If a user appears to be in crisis, self-harming, suicidal, or at risk of harming others:
- Clearly encourage contacting local emergency services, crisis hotlines, or trusted individuals immediately.
- Do not attempt therapy.

Keep responses concise (under 200 words).
"""


# wrappers/chat_service.py
async def generate_response(
    user_message: str,
    mental_state: str | None = None,
    history: list[dict] | None = None
):
    history_contents = []

    if history:
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                history_contents.append(
                    {"role": item["role"], "parts": [{"text": item["content"]}]}
                )

    prompt_text = (
        f"{SYSTEM_PROMPT}\n\nMental state: {mental_state or 'unknown'}\n\n{user_message}"
    )
    chat = client.chats.create(model=MODEL_NAME, history=history_contents)
    response = chat.send_message(prompt_text)

    return response.text