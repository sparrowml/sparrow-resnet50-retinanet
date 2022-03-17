FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH="${PATH}:/root/.poetry/bin" \
  POETRY_HTTP_BASIC_SPARROW_USERNAME=2duoZW-WIAwAQm7qwno4sYhLmYhQaztiI0 \
  POETRY_HTTP_BASIC_SPARROW_PASSWORD=""

RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt install -y \
    build-essential \
    git \
    libcairo2-dev \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9-dev python3.9-venv
RUN python3.9 -m ensurepip
RUN ln -s /usr/bin/python3.9 /usr/local/bin/python
RUN ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip
RUN pip install --upgrade pip

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false
  
COPY pyproject.toml ./
COPY poetry* ./

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=true
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --no-root --no-dev ; fi"

CMD mkdir -p /code
WORKDIR /code
