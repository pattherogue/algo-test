import subprocess
from openai import OpenAI

# Initialize OpenAI client with your API key
client = OpenAI(api_key='YOUR_API_KEY')

# Function to run the macro.py file and capture its output
def run_macro():
    result = subprocess.run(['python', 'macro.py'], capture_output=True, text=True)
    return result.stdout

# Function to interact with ChatGPT
def interact_with_gpt(input_text):
    # Use OpenAI's API to interact with ChatGPT
    response = client.completions.create(
        model="text-davinci-002",
        prompt=input_text,
        max_tokens=150  # Adjust max tokens as needed
    )
    return response.choices[0].text.strip()
    
# Run the macro.py file and capture its output
macro_output = run_macro()

# Combine the macro output with the question for ChatGPT
input_to_gpt = f"{macro_output}\n\nwhat can you tell about the state of the markets solely based on this data..."

# Interact with ChatGPT using the combined input
try:
    response = interact_with_gpt(input_to_gpt)
    print("Response from ChatGPT:", response)
except Exception as e:
    print("An error occurred:", e)
