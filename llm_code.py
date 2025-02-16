
#/// script
# requires-python = ">=3.11"
# dependencies = [
# "subprocess",
# "os",
# "validators",

#///  
import subprocess
import os
import validators

email = '23f2004637@ds.study.iitm.ac.in'
output_file = '/data/datagen_output.txt'

# Validate email format
if not validators.email(email):
    with open(output_file, 'w') as f:
        f.write('Invalid email format.')
else:
    # Run the script with the provided email as an argument
    attempts = 0
    success = False
    while attempts < 2 and not success:
        try:
            result = subprocess.run(['python3', '-c', "import requests; exec(requests.get('https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py').text)", email], capture_output=True, text=True, timeout=15)
            # Write output to the file
            with open(output_file, 'w') as f:
                f.write(result.stdout + '\n' + result.stderr)
            success = True
        except Exception as e:
            attempts += 1
            error_message = f'Attempt {attempts}: Error - {str(e)}\n'
            with open(output_file, 'a') as f:
                f.write(error_message)