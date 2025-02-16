# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip first (optional but recommended)
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies with system flag to avoid virtual environment issues
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
