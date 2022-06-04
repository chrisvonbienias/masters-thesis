FROM python:3.10.4

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv 
RUN virtualenv venv -p python3
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /masters-thesis
ADD . /masters-thesis

# dependencies
RUN pip install -r requirements.txt

# run training / testing
ARG MODE=train.py
CMD python MODE