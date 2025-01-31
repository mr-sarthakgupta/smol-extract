
        FROM pytorch/pytorch:latest
        
        RUN pip install transformers ollama pydantic fastapi uvicorn
        
        COPY model_handler.py /opt/ml/model/code/
        ENV SAGEMAKER_PROGRAM model_handler.py
        