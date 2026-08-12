# Use an official lightweight Python execution environment
FROM python:3.14-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency manifest and install required data science packages
RUN pip install --no-cache-dir numpy pandas scikit-learn

# Copy the audit framework source files into the container
COPY . /app/

# Define the default command using your exact script file name
CMD ["python", "Deployment_pipeline.py"]