import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from typing import Dict, Any
import json

class OllamaModelDeployment:
    def __init__(self, 
                 aws_role: str,
                 model_name: str = "ollama-deepseek",
                 region: str = "us-east-1"):
        self.sagemaker_session = sagemaker.Session()
        self.role = aws_role
        self.model_name = model_name
        self.region = region
        
    def build_docker_image(self) -> str:
        """Build and push Docker image to ECR"""
        ecr_client = boto3.client('ecr')
        repository_name = f"sagemaker-{self.model_name.lower()}"
        
        # Create ECR repository
        try:
            ecr_client.create_repository(repositoryName=repository_name)
        except ecr_client.exceptions.RepositoryAlreadyExistsException:
            pass
        
        # Get repository URI
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        # repository_uri = response['repositories'][0]['repositoryUri']
        repository_uri = "235494792698.dkr.ecr.us-east-1.amazonaws.com/sagemaker-ollama-deepseek"
        
        # Build and push Docker image (requires docker CLI)
        import subprocess
        dockerfile_content = """
        FROM pytorch/pytorch:latest
        
        RUN pip install transformers ollama pydantic fastapi uvicorn
        
        COPY model_handler.py /opt/ml/model/code/
        ENV SAGEMAKER_PROGRAM model_handler.py
        """
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
            
        subprocess.run(["docker", "build", "-t", repository_name, "."])
        subprocess.run(["docker", "tag", f"{repository_name}:latest", f"{repository_uri}:latest"])
        subprocess.run(["docker", "push", f"{repository_uri}:latest"])
        
        return repository_uri

    def create_model_handler(self):
        """Create model handler script"""
        handler_code = """
import json
from ollama import Client
from pydantic import BaseModel

def model_fn(model_dir):
    client = Client(host='localhost:11434')
    return client

def predict_fn(input_data, model):
    prompt = input_data.pop('prompt')
    response = model.generate(
        model='deepseek',
        prompt=prompt,
        **input_data
    )
    return response

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError('Unsupported content type')

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError('Unsupported content type')
        """
        
        with open("model_handler.py", "w") as f:
            f.write(handler_code)

    def deploy(self, 
               instance_type: str = "ml.g4dn.xlarge",
               instance_count: int = 1) -> str:
        """Deploy model to SageMaker endpoint"""
        # Build and push Docker image
        image_uri = self.build_docker_image()
        self.create_model_handler()
        
        # Create SageMaker model
        model = Model(
            image_uri=image_uri,
            role=self.role,
            name=self.model_name,
            sagemaker_session=self.sagemaker_session
        )
        
        # Deploy to endpoint
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        return predictor.endpoint_name

class SageMakerPredictor:
    def __init__(self, endpoint_name: str, region: str = "us-east-1"):
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.endpoint_name = endpoint_name
        
    def predict(self, 
                prompt: str, 
                temperature: float = 0.7,
                **kwargs) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            **kwargs
        }
        
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        return json.loads(response['Body'].read().decode())

# Usage example
if __name__ == "__main__":
    # Deploy model
    aws_role = "arn:aws:iam::235494792698:user/maayauser1"
    deployment = OllamaModelDeployment(aws_role=aws_role)
    endpoint_name = deployment.deploy()
    
    # Use deployed model
    predictor = SageMakerPredictor(endpoint_name)
    result = predictor.predict(
        prompt="What is machine learning?",
        temperature=0.7
    )
    print(result)