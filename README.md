# llm-bedrock-amazon

[![PyPI](https://img.shields.io/pypi/v/llm-bedrock-amazon.svg)](https://pypi.org/project/llm-bedrock-amazon/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/avoidik/llm-bedrock-amazon/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Amazon Titan Express model.

## Installation

Install as LLM plugin:

```terminal
$ llm install llm-bedrock-amazon
```

Install from the GitHub repository:

```terminal
$ git clone https://github.com/avoidik/llm-bedrock-amazon
$ llm install -e llm-bedrock-amazon
```

Or, from the local directory:

```terminal
$ llm install -e .
```

## Configuration

You will need to specify AWS Configuration with the normal boto3 and environment variables.

For example, to use the region `us-west-2` and AWS credentials under the `personal` profile, set the environment variables

```bash
export AWS_DEFAULT_REGION="us-west-2"
export AWS_PROFILE="personal"
```

The model must be enabled in your AWS account.

## Usage

This plugin adds the following new model:

```
amazon.titan-text-express-v1 (aliases: bedrock-titan-express, bte)
```

You can query them like this:

```terminal
$ llm -m bte "Give me 10 random names so that I can name my unnamed cat"
```

## Options

- `max_token_count`, default `4096` -- The maximum number of tokens to generate in the response.

- `temperature`, default `0.7` -- Use a lower value to decrease randomness in the response.

- `diversity`, default `0.9` -- Use a lower value to ignore less probable options and decrease the diversity of responses.
