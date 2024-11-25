> [Training](../../README.md) > [Deeplearning.ai](../README.md) > LangChain for LLM Application Development

# LangChain for LLM Application Development

- [Course Link](https://learn.deeplearning.ai/courses/langchain)

## Introduction

- None

## 01 Models Prompts and Parsers

Notebook:

- [01_model_prompt_parser.ipynb](./01_model_prompt_parser.ipynb)
- [01_model_prompt_parser.pdf](./01_model_prompt_parser.pdf)

Outline:

- `langchain.chat_models.ChatOpenAI`
- `langchain.prompts.ChatPromptTemplate`
- `langchain.output_parsers.ResponseSchema`
- `langchain.output_parsers.StructuredOutputParser`

## 02 Memory

Notebook:

- [02_langchain_memory.ipynb](./02_langchain_memory.ipynb)
- [02_langchain_memory.pdf](./02_langchain_memory.pdf)

Outline:

- `langchain.chat_models.ChatOpenAI`
- `langchain.chains.ConversationChain`
- `langchain.memory.ConversationBufferMemory`
- `langchain.memory.ConversationBufferWindowMemory`
- `langchain.memory.ConversationTokenBufferMemory`
- `langchain.memory.ConversationSummaryMemory`

```python
# LLM
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Open memory
memory = ConversationBufferMemory()

# Conversation
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```

```python
# LLM
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Memory limited to last `k` Q&A
memory = ConversationBufferWindowMemory(k=1)    

# Conversation
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```

```python
# LLM
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Memory limited to last `max_token_limit` 
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)   

# Conversation
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```

```python
# LLM
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Summary memory, allows information to effectively be "compressed"
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100) 

# Conversation
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```

![alt text](images/langchain_memory.png)

![alt text](images/langchain_memory_additional_memory_types.png)

## 03 Chains

Notebook:

- [03_langchain_chains.ipynb](./03_langchain_chains.ipynb)
- [03_langchain_chains.pdf](./03_langchain_chains.pdf)

Outline:

- `langchain.chat_models.ChatOpenAI`
- `langchain.chains.LLMChain`
- `langchain.chains.SimpleSequentialChain`
- `langchain.chains.SequentialChain`
- `langchain.prompts.ChatPromptTemplate`
- `langchain.chains.router.MultiPromptChain`
- `langchain.chains.router.llm_router.LLMRouterChain`
- `langchain.chains.router.llm_router.RouterOutputParser`

![alt text](images/langchain_chain_simple_sequential_chain.png)

![alt text](images/langchain_chain_sequential_chain.png)

![alt text](images/langchain_chain_router_chain.png)

## 04 Question & Answer

Notebook:

- [04_langchain_qa.ipynb](./04_langchain_qa.ipynb)
- [04_langchain_qa.pdf](./04_langchain_qa.pdf)

Outline:

- `langchain.chains.RetrievalQA`
- `langchain.chat_models.ChatOpenAI`
- `langchain.document_loaders.CSVLoader`
- `langchain.embeddings.OpenAIEmbeddings`
- `langchain.indexes.VectorstoreIndexCreator`
- `langchain.llms.OpenAI`
- `langchain.vectorstores.DocArrayInMemorySearch`

![alt text](images/langchain_qa_embeddings.png)

![alt text](images/langchain_qa_vector_database.png)

![alt text](images/langchain_qa_stuff_method.png)

![alt text](images/langchain_qa_additional_methods.png)

## 05 Evaluation

Notebook:

- [05_langchain_evaluation.ipynb](./05_langchain_evaluation.ipynb)
- [05_langchain_evaluation.pdf](./05_langchain_evaluation.pdf)

Outline:

- `langchain.chains.RetrievalQA`
- `langchain.chat_models.ChatOpenAI`
- `langchain.document_loaders.CSVLoader`
- `langchain.evaluation.qa.QAEvalChain`
- `langchain.evaluation.qa.QAGenerateChain`
- `langchain.indexes.VectorstoreIndexCreator`
- `langchain.vectorstores.DocArrayInMemorySearch`

## 06 Agents

Notebook:

- [06_langchain_agents.ipynb](./06_langchain_agents.ipynb)
- [06_langchain_agents.pdf](./06_langchain_agents.pdf)

Outline:

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

```python
tools = load_tools(["llm-math","wikipedia"], llm=llm)
```

```python
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
```

```python
# Define your own tool
from langchain.agents import tool

# Decorated function for agent use
@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())
```