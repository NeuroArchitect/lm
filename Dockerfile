FROM continuumio/miniconda3 as build

WORKDIR /lm

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]
RUN conda install gcc_linux-64 gxx_linux-64 make -y 

RUN pip install -U pip==20.0.2

WORKDIR /build
ADD . /build/
RUN pip install -r requirements-dev.txt 
RUN make dist


FROM continuumio/miniconda3

WORKDIR /lm

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]
RUN conda install gcc_linux-64 gxx_linux-64 make -y 

RUN pip install -U pip==20.0.2

WORKDIR /workspace/
ADD . /workspace/ 
COPY --from=build /build/dist/lm-0.1.0-py2.py3-none-any.whl .
RUN pip install lm-0.1.0-py2.py3-none-any.whl

#CMD /bin/bash
ENTRYPOINT [ "lm" ]