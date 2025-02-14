FROM mallorywittwerepfl/imaging-server-kit:latest

ARG PYTHON_VERSION=3.9
RUN conda install python=$PYTHON_VERSION -y

COPY . .

RUN python -m pip install -r requirements.txt
