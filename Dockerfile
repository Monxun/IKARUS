# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by playwright/crawl4ai for browsers
# This is a common set; you might need to adjust based on specific errors.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     # For Playwright browsers
#     libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
#     libcups2 libdrm2 libgbm1 libasound2 libx11-6 libxcomposite1 \
    # libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 \
    # libxtst6 libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
    # && apt-get clean \
    # && rm -rf /var/lib/apt/lists/*
# Commenting out apt-get for now to keep image smaller; uncomment if browser downloads fail.
# Crawl4AI handles browser downloads, but system libs might be needed.

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Playwright browsers (crawl4ai might do this on first run, but can be done here)
# RUN python -m playwright install --with-deps
# Commenting out playwright install here; crawl4ai should handle it.
# If running in a very restricted environment, pre-installing browsers might be necessary.

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (defined by PORT in .env, default 8051)
# This is informational; the actual port mapping is in docker-compose.yml
# EXPOSE ${PORT} 
# Dockerfile EXPOSE does not support variable substitution directly from .env in this way.
# The port is set by the application itself based on the PORT env var.

# Command to run the application
# The application will listen on 0.0.0.0 inside the container on the $PORT
CMD ["python", "crawl4ai_mcp.py"]
