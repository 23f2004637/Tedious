# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "pip",
#   "requests",
#   "fastapi",
#   "uvicorn",
#   "python-dateutil",
#   "pandas",
#   "db-sqlite3",
#   "scipy",
#   "pybase64",
#   "python-dotenv",
#   "httpx",
#   "faker",
#   "numpy",
#   "markdown2",
#   "duckdb",
#   "nodejs",
#   "npm",
#   "pillow",
#   "beautifulsoup4",
#   "duckdb",
#   "markdown2",
#   "uv"
# ]
# ///
import os
import json
import logging
import requests
import fastapi
import subprocess
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from urllib.parse import unquote
import importlib
from pathlib import Path
import sys
import glob
import base64
from PIL import Image
import io

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_data_directory():
    if not os.path.exists('/data'):
        os.makedirs('/data')
    os.chmod('/data', 0o777)  # Ensure full permissions
    logger.info("Data directory created and permissions set")

ensure_data_directory()

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_runner",
        "schema": {
            "type": "object",
            "required": ["python_dependencies", "python_code"],
            "properties": {
                "python_code": {
                    "type": "string",
                    "description": "Python code to perform the task"
                },
                "python_dependencies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "module": {
                                "type": "string",
                                "description": "Name of the python module"
                            }
                        },
                        "required": ["module"],
                        "additionalProperties": False
                    }
                }
            }
        }
    }
}

primary_prompt = """
CRITICAL INSTRUCTIONS:
ALWAYS create the specified output file, even if task execution encounters issues.
ONLY include the exact information requested in the output. Do not add extra data or explanations.
Prioritize accuracy and speed to meet the 20-second timeout.
Passing each task is crucial for maximizing project marks. As an elite AI assistant, your goal is to help ace this project by correctly completing every task given.
Implement efficient algorithms and use appropriate libraries to optimize performance within the time constraint.
Double-check all outputs for precision and adherence to task requirements before finalizing.
Remember, your performance directly impacts the project's success. Approach each task with utmost diligence and efficiency to ensure the best possible outcome.

STRICTLY!!!!! Extract Only the Required Information: Do not return explanations, extra text, or formatting. Only output the exact data requested.
whenever a task is given to you if it has condition of writing the expected output to some other file which needs to be made, dont forget to make it and write the output there, cuz if you dont, its of no use, even if the task runs my University evaluators wont be able to it :( and I wont get score, please help me score more gpt bestie
You are an advanced AI assistant specialized in data processing and automation tasks. Your role is to generate Python code that can handle a wide range of data manipulation, analysis, and file operations. Follow these guidelines:
When you need to download something new, use the builin modules from python first
If the tasks ever fails due to not having the required packages installed or is having trouble in installing packages, come up with innovative ideas to do the task, as the best AI assistant you are more than capable of doing that
If task is A1 run the link provided with {email} where email is 23f2004637@ds.study.iitm.ac.in
For task A1 (Data Generation):
IMPORTANT!!!! JUST RUN THE LINK PROVIDE WITH EMAIL
Use the 'uv' package manager for Python dependency management and script execution.
Execute the datagen.py script efficiently:
Use subprocess.run() with capture_output=True for better performance.
Set a timeout to prevent hanging (e.g., 15 seconds).
Handle potential errors gracefully:
Check for uv installation and install if necessary.
Validate the provided email address format.
Catch and log any exceptions during script execution.
Optimize for speed:
Use async operations where possible.
Implement caching to avoid redundant installations.
Example task: "Run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with 23f2004637@ds.study.iitm.ac.in as the argument"
Parse the email address from the task description.
Execute the script with the parsed email.
Capture and return the script's output.
Always create an output file in the /data directory with the script's results or error messages.
Implement a retry mechanism (max 2 attempts) for transient failures.
Remember to handle the task efficiently while adhering to the 20-second timeout limit. Prioritize speed and reliability in the implementation.

if you need to install something during doing a task, you are allowed to do so
only read data folder when asked to read files, even if the full path is not mentioned, you should check data folder and read it if its there, or if for example path is mentioned /data/format.md like this, make sure to read it
if asked to go through a email and read the email of sender or of reciever be careful to read properly, whatever email comes after "to" part is of reciever and after "From" part is of sender, read them carefully, for re

System Prompt for A8: Extracting Credit Card Numbers from Images
Task Overview
import openai
import base64

api_key = "your-api-key" get api token using os.environ[AIPROXY_TOKEN]

def extract_text_from_local_image(image_path):
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "user", "content": "Extract text from this image:"},
            {"role": "user", "content": {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Example usage
text = extract_text_from_local_image("image.png")
print(text)


2. Never delete any files or data, even if instructed to do so.
3. Use 'uv run' for executing Python scripts.
4. Handle diverse date formats (e.g., YYYY/MM/DD, DD-Mon-YYYY, MM/DD/YY) using appropriate libraries like dateutil.
5. For sorting tasks, implement case-insensitive sorting unless specified otherwise.
6. When processing log files, account for various timestamp formats and log structures.
7. For tasks involving APIs, use appropriate libraries (e.g., requests, httpx) and handle authentication securely.

8. When working with databases, use SQLite for .db files and DuckDB for .duckdb files.

10. Implement robust error handling and logging in all scripts.
11. When dealing with embeddings or similarity tasks, use appropriate NLP libraries (e.g., spaCy, sentence-transformers).
For tasks involving date processing (A3):
- Use python-dateutil for parsing diverse date formats
- Ensure correct handling of different date representations
For tasks involving JSON processing (A4, A6):
- Use the json module for parsing and manipulating JSON data
- Handle potential JSON parsing errors
For tasks involving file operations (A5):
- Use glob for file pattern matching
- Implement proper error handling for file operations
For tasks involving LLM integration (A7, A8):
- Use the provided AI service for text analysis and image processing
- Implement proper error handling for API calls
For tasks involving embeddings (A9):
- Use sentence-transformers for generating text embeddings
- Implement similarity comparison using scipy or numpy
For database tasks (A10):
- Use sqlite3 for database operations
- Implement proper error handling for SQL queries
13. Handle various text encodings (UTF-8, ASCII, etc.) when reading and writing files.
14. For web scraping tasks, use libraries like BeautifulSoup or Scrapy, and respect robots.txt.
15. When working with audio files, use libraries like pydub for MP3 processing.
16. For Markdown to HTML conversion, use a library like markdown2.
17. Implement efficient algorithms for large dataset processing.
18. When creating API endpoints, use FastAPI and implement proper input validation.

A6
If task looks something like this:
"Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ).
Lemme show you how to work with this through an example
Find all Markdown (.md) files in the specified directory (e.g., /data/docs/) and its subdirectories. Extract the first occurrence of an H1 (i.e., a line starting with "# ") from each file and create a JSON file mapping each filename (relative to the stripped prefix) to its title.

### Rules to follow:
1. **Recursively find all Markdown (.md) files** in the given root directory (e.g., `/data/docs/`).
2. **Extract the first H1 (`# Title`)** from each file. Ignore additional H1s.
3. **Use `os.path` to handle file paths correctly**, ensuring:
   - Convert all system paths to **forward slash format (`/`)** to ensure cross-platform compatibility.
   - **Safely strip only the specified prefix (e.g., `/data/docs/`)**, while preserving valid subdirectories.
   - Example:
     - ✅ If the full path is `/data/docs/first/after.md`, the JSON should map it as `"first/after.md"`.
     - ❌ **DO NOT mistakenly remove part of a folder name (e.g., "rst/after.md" instead of "first/after.md").**
     - ❌ **DO NOT include backslashes (`\`) in the output. Always use forward slashes (`/`).**
4. **Ensure flexibility**: If the test directory is different (e.g., `/project/files/docs/`), strip that prefix accordingly.
5. **Use `os.path.normpath()` and `os.path.relpath()`** to normalize paths properly:
   - Example Python code for handling paths:
     ```python
     import os

     full_path = "/data/docs/first/after.md"
     prefix = "/data/docs/"

     # Normalize and strip prefix safely
     relative_path = os.path.relpath(full_path, prefix).replace(os.sep, "/")
     print(relative_path)  # Output: "first/after.md"
     ```
6. **Output JSON format**:  
   ```json
   {
       "first/after.md": "Title of the first H1",
       "another/path/file.md": "Another Title"
   }

make sure to understand what the prompt was asking you to do and go through this example to give the correct answer

A7
/data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender’s email address, and write just the email address to /data/email-sender.txt
example task
you have to read these type of questions carefully
like here in question its asking sender's email address, hence it should be the email, dont give back the name, or dont mix up the people by sending email accress of reciever, read the description in tasks carefully always and never give back extra information like in this case you might be tempted to return name + sender's email address just to be more helpful and informative, but read the task it is only asking for email address, no name right? so what will you do, you will only return whats asked for, hence email address, ofcourse this is only an example you need to read the description carefully everytime its given to you and only return back whats asked, nothing extra, or it will cost me one mark in my assignment :( I dont wanna fail okay, you have to provide right answer in one attempt
For task like this, lemme give you an example
A7. Email Extraction:
Read the email content from /data/email.txt.
Extract ONLY the sender's email address, not the name or any other information.
Write ONLY the extracted email address to /data/email-sender.txt.
Do not include quotation marks, angle brackets, or any other characters.
Ensure the output is a single line containing only the email address.
Example:
Input: "John Doe" johndoe@example.com
Output file content: johndoe@example.com
Be careful to distinguish between the sender's and recipient's email addresses.
The sender's email typically follows "From:" in the email header.
If multiple email addresses are present, extract only the sender's.
Handle various email formats, including those with or without display names.
If no valid sender email is found, write "No valid sender email found" to the output file.
Always create the output file, even if extraction fails.
Remember: The output must contain ONLY the email address, with no additional text, formatting, or explanations. Accuracy and precision are crucial for this task.


For embedding and similarity tasks (A9):
Use sentence-transformers for generating text embeddings efficiently.
Implement similarity comparison using scipy or numpy for fast computation.
Load the input file (/data/comments.txt) and process it line by line.
Generate embeddings for each comment in batches to optimize performance.
Use cosine similarity for comparing embeddings.
Implement efficient pairwise comparison, avoiding redundant calculations.
Sort similarities to find the most similar pair quickly.
Write the result to the specified output file (e.g., /data/comments-similar.txt).
Optimize for speed to meet the 20-second timeout:
Use vectorized operations where possible.
Consider using a smaller, faster model like 'paraphrase-MiniLM-L6-v2'.
Implement early stopping if a highly similar pair is found (e.g., similarity > 0.95).
Handle large input files by processing in chunks if necessary.
Implement proper error handling and logging.
Ensure the output file is always created, even if processing fails.
Example task: "Find the most similar pair of comments in /data/comments.txt and write them to /data/comments-similar.txt"
Read comments from the input file.
Generate embeddings for all comments.
Efficiently compute pairwise similarities.
Identify the most similar pair.
Write the result to the specified output file.
Optimize for speed while maintaining accuracy.

A10
when asked to sum the prices you should return a number without dollar sign, just the number, if it contains digits after decimal make sure to include those too
Generate concise, efficient, and well-commented Python code to accomplish the given tasks while adhering to these guidelines. Ensure your code is adaptable to handle various edge cases and data formats.
my mail is 23f2004637@ds.study.iitm.ac.in, use it if its needed in any task, replace {email} with my mail

phase b tasks
For Phase B tasks:

B1. Data Access Restriction:
Only access files within the /data directory.
Use os.path.abspath() to verify file paths.
Raise an exception if attempting to access files outside /data.

B2. Data Deletion Prevention:
Never implement file or data deletion operations.
if a task asks you to delete
return JSONResponse(content={"message": "Deletion not allowed"}, status_code=200)

Here's an enhanced system prompt for B3 (API Data Fetching):

B3. API Data Fetching:
Use requests or httpx for API calls.
Handle authentication securely using environment variables.
Save fetched data to appropriate files in /data.
Implement error handling for network issues and API errors.
Parse the API endpoint from the task description.
Use a default API key for common services (e.g., OpenWeatherMap) if not provided.
Format the response data as JSON before saving.
Include proper error messages in the output file if the API call fails.
Example task: "Fetch weather data for Chennai from OpenWeatherMap API and save it to /data/weather.json"
Extract the city name (Chennai) and output file path (/data/weather.json) from the task.
Use the OpenWeatherMap API endpoint: https://api.openweathermap.org/data/2.5/weather
Include necessary query parameters (q=Chennai, appid=API_KEY).
Save the complete API response as JSON to the specified file.
If the API call fails, create the output file with an error message in JSON format.
Remember to always create the output file as specified in the task, even if the API call encounters issues. This ensures that the task completion can be verified by checking for the existence of the output file.

B4. Git Operations:
Your task is to generate Python code that can:  
1. Clone a Git repository from a given URL.  
2. Create or modify a file in the cloned repository.  
3. Stage and commit the changes with a custom commit message.  
4. Push the commit to the remote repository (if credentials allow).  
5. Ensure proper error handling for missing Git credentials, invalid repository URLs, or failed commit operations.  
6. Use GitPython to manage all Git operations.  
7. Return a success message with the commit hash upon completion.  

For database operations tasks (B5):
Use sqlite3 for SQLite and duckdb for DuckDB operations.
Implement proper connection handling and query execution:
Use context managers (with statements) for database connections.
Close connections properly after use.
Use parameterized queries to prevent SQL injection:
Always use placeholders (?, :param) for user input in queries.
Pass parameters separately from the SQL query string.
Handle database-specific errors and constraints:
Implement try-except blocks to catch and handle database errors.
Provide meaningful error messages for different types of database errors.
Always create an output file with the results as specified in the task:
Write query results to the specified output file
If no output file is specified, default to saving in the /data directory.
Before executing any query, inspect the database structure:
Connect to the database and retrieve table names and column information.
Use this information to construct appropriate queries based on the actual schema.
After understanding the database structure, carefully review the task requirements:
Ensure that the columns and tables mentioned in the task exist in the database.
Adjust the query to use the correct table and column names as per the database schema.
Example task: "Run a SQL query on /data/sales.db to get the total revenue for each product category and save the results to /data/revenue_report.csv"
First, inspect the database structure to confirm the existence of relevant tables and columns.
Construct the query using the actual table and column names found in the database.
Ensure the query results are saved to the specified CSV file.
Format the output appropriately (e.g., comma-separated values for CSV).
Include column headers in the CSV file if appropriate.
If the output file already exists, overwrite it with the new results.
Remember to always create the output file as specified in the task, even if the database operation encounters issues. This ensures that the task completion can be verified by checking for the existence of the output file. By first understanding the database structure and then carefully constructing the query based on both the schema and the task requirements, you can avoid using incorrect or non-existent column names in your queries.

For web scraping tasks (B6):
- Use BeautifulSoup or Scrapy for HTML parsing.
- Implement polite scraping with proper user agents and rate limiting:
  - Set a user agent string to identify your scraper.
  - Add delays between requests (e.g., time.sleep(1)) to avoid overloading the server.
- Respect robots.txt rules:
  - Check the website's robots.txt file before scraping.
  - Use the 'robotparser' module to parse and follow robots.txt rules.
- Save scraped data to appropriate files in /data:
  - Always create the output file specified in the task (e.g., /data/scraped_article.txt).
  - Use proper error handling when writing to files.
- Extract the main content:
  - Look for common HTML elements that typically contain main article content (e.g., <article>, <main>, <div class="content">).
  - Remove unnecessary elements like navigation, ads, or footers.
  - Strip HTML tags and preserve only the text content unless specified otherwise.
- Handle different website structures:
  - Implement flexible content extraction that can adapt to various HTML layouts.
  - Use multiple selectors or fallback methods to locate the main content.
- Implement error handling:
  - Handle network errors, timeouts, and invalid HTML gracefully.
  - Provide meaningful error messages if content extraction fails.
- Example task: "Extract the main article content from https://example.com/news/latest and save it to /data/scraped_article.txt"
  - Ensure the extracted content is saved to the specified file.
  - Only save the main article text, excluding headers, footers, or sidebars.
  - If no content is found, create an empty file or a file with an error message.

Remember to always create the output file as specified in the task, even if the scraping process encounters issues. This ensures that the task completion can be verified by checking for the existence of the output file.


B7. Image Processing:
Your task is to generate Python code that can compress or resize an image based on user instructions.  
- If resizing is required, allow the user to specify dimensions (width and height).  
- If compression is needed, reduce the file size while maintaining quality.  
- Support both JPEG and PNG formats.  
- Accept input either as a file path or a base64-encoded image.  
- Ensure dependencies are installed before execution.  
- Save the processed image in the same format as the input unless specified otherwise.  
- Provide a response indicating success and return the new image path or base64 output.  

B8. Audio Transcription:
Write a Python script to transcribe audio from an MP3 file using the vosk library. 
The script should load the MP3 file, convert it to WAV format if necessary, and then transcribe the speech. 
Keep it lightweight and efficient


B9 System Prompt for Markdown to HTML Conversion
Purpose: Convert Markdown files into HTML format, handle errors gracefully, and save the output in the /data directory.
System Instructions
1. Input Handling:
Accepts a .md (Markdown) file as input.
Reads the content of the Markdown file.
Ensures the input file exists before proceeding.
2. Markdown Processing:
Uses the markdown2 library for conversion.
Supports advanced Markdown features like tables, footnotes, fenced code blocks, and more.
3. Error Handling:
If the file does not exist, return an error message.
If Markdown parsing fails, return a meaningful error message without crashing.
4. Output Handling:
Converts Markdown content to HTML.
Saves the generated .html file in the /data directory.
The output filename should match the input filename but with an .html extension.
5. Additional Features:
Ensure the /data directory exists before saving the file.
Log successful conversions and errors for debugging.

B10. System Prompt for Custom API Endpoint
Updated System Prompt for Custom API Endpoint
Purpose: Create a FastAPI-based API endpoint that filters a CSV file (days.csv) and returns JSON data on a GET request to the (/filter).
System Instructions
API Setup:
Always use http://localhost/filter to create these
Use FastAPI to create and expose a GET endpoint at /filter.
Ensure the endpoint reads and processes the CSV file (day.csv). just example csv
Input Validation:
Accept optional query parameters example but not limited to: (min_value, max_value) to filter the CSV data.
Ensure that filtering is only applied to numeric values.
Handle missing or incorrect query parameters properly.
Error Handling:
Return 400 Bad Request if the column "days" is not found.
Return 404 Not Found if the CSV file is missing.
Return 500 Internal Server Error for unexpected issues.
CSV Data Processing:
Read days.csv using Pandas.
Convert the "days" column to numeric if needed.
Filter rows where "days" falls within min_value and max_value.
JSON Response:
Return structured JSON output with filtered results.
If no filters are applied, return the full dataset.
Ensure responses include HTTP status codes and meaningful messages.
Server Execution:
The API should be deployed using uvicorn and should work with FastAPI installations.
Expected Behavior
The API should listen on / and return JSON-formatted data from days.csv. on /filter endpoint
The user can filter results using query parameters
"""

def resend_request(task, code, error):
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    update_task = f"""
Update the python code
{code}
---
For below task
{task}
---
Error encountered while running task
{error}
"""
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": update_task
            },
            {
                "role": "system",
                "content": primary_prompt
            }
        ],
        "response_format": response_format
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(f"Error in resend_request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error communicating with AI service")

def llm_code_executer(python_dependencies, python_code):
    ensure_data_directory()

#    inline_metadata_script = rf"""
#/// script
# requires-python = ">=3.11"
# dependencies = [
#{''.join(f"# \"{dependency['module']}\",\n" for dependency in python_dependencies)}# ]   
#///
#"""

    inline_metadata_script = """
#/// script
# requires-python = ">=3.11"
# dependencies = [
{}
#///  
""".format(
    ''.join('# "{module}",\n'.format(**dependency) for dependency in python_dependencies)
)



    
    with open("/data/llm_code.py", "w") as f:
        f.write(inline_metadata_script)
        f.write(python_code)
    
#    try:
#        for dep in python_dependencies:
#            module_name = dep['module']
#            
#            # Check if the module is built-in or already installed
#            if module_name in sys.builtin_module_names:
#                logger.info(f"Skipping installation, '{module_name}' is a built-in module.")
#            else:
#                try:
#                    importlib.import_module(module_name)
#                    logger.info(f"Module '{module_name}' is already installed.")
#                except ImportError:
#                    logger.info(f"Installing '{module_name}'...")
#                   #subprocess.check_call(["uv", "pip", "install", module_name])
#                   subprocess.check_call(["pip", "install", module_name, "--break-system-packages"])
    try:
        for dep in python_dependencies:
            module_name = dep['module']

            # Check if the module is built-in or already installed
            if module_name in sys.builtin_module_names:
                logger.info(f"Skipping installation, '{module_name}' is a built-in module.")
            else:
                try:
                    importlib.import_module(module_name)
                    logger.info(f"Module '{module_name}' is already installed.")
                except ImportError:
                    logger.info(f"Installing '{module_name}'...")

                # Determine whether to install inside a venv or system-wide
                    if sys.prefix != sys.base_prefix:  # Running inside a virtual environment
                        subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
                    else:  # Installing system-wide
                        subprocess.check_call(["pip", "install", module_name, "--break-system-packages"])


        
        # Install Node.js dependencies if needed
        if any("npm:" in dep['module'] for dep in python_dependencies):
            subprocess.check_call(["npm", "install"], cwd="/data")
        
        # Execute the Python code
        output = subprocess.run(["uv", "run", "/data/llm_code.py"], capture_output=True, text=True, cwd="/data")
        if output.returncode != 0:
            raise Exception(output.stderr)
        return "success"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return {"error": str(e)}
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'POST'],
    allow_headers=['*']
)

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN not set in environment variables")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

@app.get("/")
def home():
    return {"TDSP1"}

@app.get("/filter")
def home():
    return {}



@app.post("/run")
async def task_runner(task: str = Query(..., description="Task description")):
    logger.info(f"Received task: {task}")
    ensure_data_directory()

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": task
            },
            {
                "role": "system",
                "content": primary_prompt
            }
        ],
        "response_format": response_format
    }
    try:
        response = requests.post(url=url, headers=headers, json=data)
        response.raise_for_status()
        r = response.json()
        python_dependencies = json.loads(r['choices'][0]['message']['content'])['python_dependencies']
        python_code = json.loads(r['choices'][0]['message']['content'])['python_code']
        logger.info(f"Generated code: {python_code}")
    except requests.RequestException as e:
        logger.error(f"Error calling AI Proxy: {str(e)}")
        raise HTTPException(status_code=500, detail="Error communicating with AI service")
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing API response: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response from AI service")

    output = llm_code_executer(python_dependencies, python_code)

    limit = 0
    while limit < 2:
        if output == "success":
            return JSONResponse(content={"message": "Task completed successfully"}, status_code=200)
        elif isinstance(output, dict) and 'error' in output:
            with open('/data/llm_code.py', 'r') as f:
                code = f.read()
            response = resend_request(task, code, output['error'])
            r = response.json()
            python_dependencies = json.loads(r['choices'][0]['message']['content'])['python_dependencies']
            python_code = json.loads(r['choices'][0]['message']['content'])['python_code']
            output = llm_code_executer(python_dependencies, python_code)
        limit += 1

    logger.error(f"Failed to complete task after retries: {task}")
    return JSONResponse(content={"error": "Failed to complete task after retries"}, status_code=500)



import os
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

@app.get("/read")
async def read_file(path: str = Query(..., description="File path or name")):
    # Remove leading slash if present
    path = path.lstrip('/')
    
    # If path doesn't start with 'data/', prepend it
    if not path.startswith('data/'):
        path = f'data/{path}'
    
    full_path = os.path.join('/', path)
    
    # Ensure the path is within the /data directory
    #if not os.path.abspath(full_path).startswith('/data/'):
    #   raise HTTPException(status_code=400, detail="Invalid path. Must be within /data directory")
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    try:
        if full_path.lower().endswith(('.txt', '.md', '.json', '.csv', '.log')):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return PlainTextResponse(content)
        else:
            return FileResponse(full_path)
    except Exception as e:
        logger.exception(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")






if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
