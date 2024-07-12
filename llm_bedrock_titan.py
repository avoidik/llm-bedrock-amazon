"""LLM plugin to invoke AWS Amazon Titan model in Bedrock."""

from typing import Iterator, Optional

import json
import boto3
import llm
from pydantic import Field, field_validator

# Much of this code is derived from:
# - https://github.com/tomviner/llm-claude
# - https://github.com/sblakey/llm-bedrock-anthropic


@llm.hookimpl
def register_models(register):
    """Register model in LLM."""
    register(
        BedrockTitan("amazon.titan-text-express-v1"),
        aliases=("bedrock-titan-express", "bte"),
    )


class BedrockTitan(llm.Model):
    """AWS Amazon Titan model in Bedrock."""
    can_stream: bool = True
    model_id: str

    class Options(llm.Options):
        """Configuration of AWS Amazon Titan model."""
        max_token_count: Optional[int] = Field(
            description="The maximum number of tokens to generate in the "
                        "response.",
            default=4096,
        )

        temperature: Optional[float] = Field(
            description="Use a lower value to decrease randomness in the "
                        "response.",
            default=0.7,
        )

        diversity: Optional[float] = Field(
            description="Use a lower value to ignore less probable options "
                        "and decrease the diversity of responses.",
            default=0.9,
        )

        @field_validator("max_token_count")
        @classmethod
        def validate_max_token_count_length(cls, max_token_count):
            """Check max_token_count length."""
            if not (0 < max_token_count <= 8192):
                raise ValueError("max_token_count must be in range 1-8192")
            return max_token_count

        @field_validator("temperature")
        @classmethod
        def validate_temperature_length(cls, temperature):
            """Check temperature length."""
            if not (0.0 < temperature <= 1.0):
                raise ValueError("temperature must be in range 0.0-1.0")
            return temperature

        @field_validator("diversity")
        @classmethod
        def validate_diversity_length(cls, diversity):
            """Check diversity length."""
            if not (0.0 < diversity <= 1.0):
                raise ValueError("diversity must be in range 0.0-1.0")
            return diversity

    def __init__(self, model_id):
        self.model_id = model_id
        self.bedrock = boto3.client(service_name="bedrock-runtime")

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation],
    ) -> Iterator[str]:
        if prompt.system:
            prompt.prompt = prompt.system + "\n" + prompt.prompt

        body = {
            "inputText": prompt.prompt,
            "textGenerationConfig": {
                "maxTokenCount": prompt.options.max_token_count,
                "temperature": prompt.options.temperature,
                "topP": prompt.options.diversity,
                "stopSequences": [],
            }
        }

        chunks = []
        if stream:
            for chunk in self.generate_text_streaming(json.dumps(body)):
                chunks.append(chunk)
                yield chunk
        else:
            response_body = self.generate_text(json.dumps(body))
            for chunk in response_body:
                chunks.append(chunk)
                yield chunk

        response.response_json = {"messages": chunks}

    def generate_text_streaming(self, body):
        """Invoke AWS Titan Model with the streaming support."""
        response = self.bedrock.invoke_model_with_response_stream(
            accept="application/json",
            contentType="application/json",
            body=body,
            modelId=self.model_id,
        )

        response_body = response.get("body")
        for event in response_body:
            chunk = json.loads(event["chunk"]["bytes"])
            if "outputText" in chunk:
                yield chunk["outputText"]

    def generate_text(self, body):
        """Invoke AWS Titan Model without the streaming support."""
        response = self.bedrock.invoke_model(
            accept="application/json",
            contentType="application/json",
            body=body,
            modelId=self.model_id,
        )

        model_response = json.loads(response["body"].read())
        for result in model_response["results"]:
            yield result["outputText"]
