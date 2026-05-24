import ollama

MODEL_NAME = 'llama3'

def classify_email_zero_shot(email_text):
    # System instructions keep the model strictly limited to single-word responses
    system_instruction = (
        "You are an AI that only outputs exactly one word. "
        "You must choose from: Work, Personal, or Spam. "
        "Do not include periods, explanations, or introductory text."
    )
    
    # Clean zero-shot prompt with no training examples
    prompt = f"""
Classify the following email into one of these categories: Work, Personal, or Spam.

Email: "{email_text}"
Category:"""

    response = ollama.generate(
        model=MODEL_NAME, 
        system=system_instruction,
        prompt=prompt,
        options={"temperature": 0.0}
    )
    
    return response['response'].strip()


# The full list of all 5 emails from your instructions
all_emails = [
    "Don't forget the team meeting at 2 PM. Please bring your project updates.",
    "Dinner at 8 tonight? I’ll bring the wine.",
    "You have won a free iPhone! Click here to claim your prize.",
    "The Q2 financial report is due by end of day tomorrow.",
    "Are you free for lunch this weekend?"
]


if __name__ == "__main__":
    print("--- Running Full Zero-Shot Classification ---")
    
    for i, email in enumerate(all_emails, 1):
        output = classify_email_zero_shot(email)
        print(f"Email #{i}: \"{email}\"")
        print(f"Result:   {output}")
        print("-" * 50)