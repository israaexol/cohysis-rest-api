#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import json
from joblib import load
from transformers import BertTokenizer
from BERTSem import BERTSem

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch

from predict import predict, preprocess
from config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *

# Initialize API Server
app = FastAPI(
    title="Polysis",
    description="Evaluate your discourse coherence using the one and only BERT!",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static folder, like demo pages, if any
app.mount("/static", StaticFiles(directory="static/"), name="static")

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

with open("config.json") as json_file:
    model_config = json.load(json_file)

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    tokenizer = BertTokenizer.from_pretrained(model_config["BERT_MODEL"])
    # Initialize the pytorch model
    model = BERTSem()
    model.load_state_dict(torch.load(
        CONFIG['MODEL_PATH'], map_location=torch.device('cpu')))
    model = model.eval()

    # add model and other preprocess tools too app state
    app.package = {
        "tokenizer": tokenizer,
        "model": model
    }


@app.post('/evaluate',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_predict(request: Request, body: InferenceInput):
    """
    Perform prediction on input data
    """

    logger.info('API predict called')
    logger.info(f'input: {body}')

    # prepare input data
    text = body.text
    print(text)
    data_loader = preprocess(app.package, text=text)
    # run model inference
    pred = predict(app.package, data_loader=data_loader)

    # prepare json for returning
    results = {
        'pred': pred
    }

    logger.info(f'results: {results}')

    return {
        "error": False,
        "results": results
    }


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

@app.get("/")
def root():
    return {"data": "Welcome to Polysisssss"}

if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True, debug=True, log_config="log.ini"
                )