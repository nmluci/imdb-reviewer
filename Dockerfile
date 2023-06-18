FROM tensorflow/tensorflow:2.12.0
LABEL maintainer="Lynne Fuyuna"
LABEL build_date="2023-06-18"
LABEL description="sentichan. an ai model for emotional sentiment analysis"

WORKDIR /app

ADD . .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]