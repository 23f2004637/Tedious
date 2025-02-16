import os
import requests
import subprocess

def execute_a1_task(url, email):
    # Ensure /data directory exists
    os.makedirs('/data', exist_ok=True)
    
    # Download the script
    script_path = '/data/datagen.py'
    response = requests.get(url)
    response.raise_for_status()
    
    with open(script_path, 'wb') as f:
        f.write(response.content)
    
    # Execute the script
    try:
        result = subprocess.run(['python', script_path, email], 
                                capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return "Data generation completed successfully"
        else:
            return f"Error in data generation: {result.stderr}"
    except Exception as e:
        return f"Error executing script: {str(e)}"