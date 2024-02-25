import subprocess
from openai import OpenAI

client = OpenAI(api_key='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')

# Function to run the macro.py file and capture its output
def run_macro():
    result = subprocess.run(['python', 'macro.py'], capture_output=True, text=True)
    return result.stdout

# Function to interact with ChatGPT
def interact_with_gpt(input_text):
    # Use OpenAI's API to interact with ChatGPT
    # Replace 'YOUR_API_KEY' with your actual API key
    response = client.completions.create(
    model="text-davinci-002",
    prompt=input_text,
    max_tokens=150  # Adjust max tokens as needed)
    )
    return response.choices[0].text.strip()
    
# Run the macro.py file and capture its output
macro_output = run_macro()

# Combine the macro output with the question for ChatGPT
input_to_gpt = f"{macro_output}\n\nwhat can you tell about the state of the markets solely based on this data. the sector rotation. or if you were a trader. make sure you look at everything in relation to each other. you don't need to explain your analysis step by step in detail only the outcome of your analysis. just really brief. make sure you look at every detail. we want to look at correlations between indicators, etc. and then we want to estimate what phase of the business cycle. and its only after we've done all of this do we want to look for the sector rotation and forward guidance. we want to account for all of the nuance being displayed here and get the most robust assessment of market health possible:"

# Interact with ChatGPT using the combined input
response = interact_with_gpt(input_to_gpt)

# Print the response from ChatGPT
print("Response from ChatGPT:", response)
