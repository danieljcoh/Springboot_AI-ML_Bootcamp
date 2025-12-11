# Tells a server how to build your app.

# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY main.py .
COPY model_utils.py .
COPY pokemon.csv .
COPY pokemon_battle_model.pkl .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]