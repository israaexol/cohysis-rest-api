#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    text: str = Field(..., example='John ate an apple', title='Text to analyze')


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    pred: str = Field(..., example='Medium', title='Predicted class with highest probablity')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Whether there is error')
    results: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')