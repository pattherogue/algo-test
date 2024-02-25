import subprocess
import pyautogui
import time

# Step 1: Determine the path to your_script.py within the algo-test directory
script_path = "macro.py"  # Assuming your_script.py is located within the algo-test directory

# Step 2: Run your Python script and capture its output
result = subprocess.run(["python", script_path], capture_output=True, text=True)
output = result.stdout

# Step 3: Simulate typing the output into ChatGPT
pyautogui.write(output)

# Step 4: Simulate pressing Enter
pyautogui.press('enter')

# Step 5: Add a small delay before typing the analysis prompt
time.sleep(1)

# Analysis prompt
prompt = """
"what can you tell about the state of the markets solely based on this data. 
the sector rotation. or if you were a trader. make sure you look at everything 
in relation to each other. you don't need to explain your analysis step by step 
in detail only the outcome of your analysis. just really brief. make sure you look 
at every detail. we want to look at correlations between indicators, etc. and then 
we want to estimate what phase of the business cycle. and its only after we've done 
all of this do we want to look for the sector rotation and forward guidance. we want 
to account for all of the nuance being displayed here and get the most robust 
assessment of market health possible:"
"""

# Step 6: Simulate typing the analysis prompt
pyautogui.write(prompt)

# Step 7: Simulate pressing Enter again to submit the prompt
pyautogui.press('enter')
