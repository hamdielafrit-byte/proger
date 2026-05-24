from flask import Flask, render_template, request, jsonify
import ollama

app = Flask(__name__)

# We use translategemma, but you can swap this to 'llama3.2' or 'qwen2.5' if needed
MODEL_NAME = 'translategemma'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text_to_translate = data.get('text', '')
    source_lang = data.get('source_lang', 'English')
    target_lang = data.get('target_lang', 'Spanish')

    if not text_to_translate.strip():
        return jsonify({'translated_text': ''})

    # The ideal prompt format to force local LLMs to ONLY return translation text
    system_prompt = (
        f"You are a professional {source_lang} to {target_lang} translator. "
        f"Accurately convey the meaning and nuances of the text while adhering to "
        f"{target_lang} grammar and cultural sensitivities. "
        f"Output ONLY the raw translation. Do not include introductory notes, explanations, "
        f"or quotation marks around the response."
    )
    
    user_prompt = f"Please translate the following text into {target_lang}:\n\n{text_to_translate}"

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': 0.1 # Keeps the output stable, accurate, and repeatable
            }
        )
        
        translated_text = response['message']['content'].strip()
        return jsonify({'translated_text': translated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)