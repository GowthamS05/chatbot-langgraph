FROM python:3.12-slim

WORKDIR /main

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:main", "--host", "0.0.0.0", "--port", "8000"]