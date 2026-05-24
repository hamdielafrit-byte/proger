import ollama

MODEL_NAME = 'llama3'

def classify_email(email_text):
    # System instructions force the model to behave and not chat
    system_instruction = (
        "You are an AI that only outputs exactly one word. "
        "You must choose from: Work, Personal, or Spam. "
        "Do not include periods, explanations, or introductory text."
    )
    
    # The few-shot prompt structure (Contains the 3 core examples)
    prompt = f"""
Classify each email into one of the following categories: Work, Personal, or Spam.

Email: "Dinner at 8 tonight? I’ll bring the wine."  
Category: Personal
 
Email: "You have won a free iPhone! Click here to claim your prize."  
Category: Spam
 
Email: "The Q2 financial report is due by end of day tomorrow."  
Category: Work
 
Email: "{email_text}"  
Category:"""

    try:
        # Pass both system instructions and prompt to Ollama
        response = ollama.generate(
            model=MODEL_NAME, 
            system=system_instruction,
            prompt=prompt,
            options={
                "temperature": 0.0  # Completely predictable and stable
            }
        )
        return response['response'].strip()
        
    except Exception as e:
        # Safeguard if Ollama background runners crash
        return f"ERROR (Local runner error: {str(e)})"


# Complete list of all 5 emails from your instructions
all_emails = [
    "Don't forget the team meeting at 2 PM. Please bring your project updates.",
    "Dinner at 8 tonight? I’ll bring the wine.",
    "You have won a free iPhone! Click here to claim your prize.",
    "The Q2 financial report is due by end of day tomorrow.",
    "Are you free for lunch this weekend?"
]


if __name__ == "__main__":
    print("--- Running Full Few-Shot Email Classification ---")
    print("Each item below is run through the model one by one:\n")
    
    for i, email in enumerate(all_emails, 1):
        output = classify_email(email)
        print(f"Email #{i}: \"{email}\"")
        print(f"Output:   {output}")
        print("-" * 50)