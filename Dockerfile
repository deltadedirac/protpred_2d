FROM mapler/pytorch-cpu:latest

RUN apt-get update && apt-get install nano wget
WORKDIR /protpred_2D
COPY . /protpred_2D

#docker run --name protpred_2d -it -v ${PWD}:/protpred_2D/ protpred_2d /bin/bash
