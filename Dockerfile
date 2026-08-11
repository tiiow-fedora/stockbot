FROM python:3.11-slim

WORKDIR /app

# System deps needed to build psycopg2 from source if the binary wheel
# isn't available for the target platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home stockbot
USER stockbot

CMD ["python", "main.py", "run"]
