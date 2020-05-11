
FROM pytorch/pytorch:latest

RUN apt update \
    && apt install -y \
        htop \
    && rm -rf /var/lib/apt/lists/*

ENV PORT=8899
ENV PROJECT_ROOT /app
ENV DATA_PATH /datasets

WORKDIR $PROJECT_ROOT
RUN mkdir -p $DATA_PATH

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN rm requirements.txt

EXPOSE ${PORT}

# CMD ["bin", "bash"]
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=${PORT}", "--no-browser", "--allow-root"]