FROM continuumio/miniconda3

WORKDIR /lm

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]
RUN conda install gcc_linux-64 gxx_linux-64 make -y 

ADD requirements-dev.txt .
RUN pip install -U pip==20.0.2
RUN pip install -r requirements-dev.txt 

CMD /bin/bash
