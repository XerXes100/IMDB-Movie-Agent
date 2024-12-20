# IMDB Movie Agent using Langchain and OpenAI

## Overview
This project demonstrates the use of Langchain, OpenAI’s GPT-4 model, and a dataset of the top 1000 IMDb movies to build an intelligent agent capable of answering movie-related queries. The agent is powered by Langchain’s `create_pandas_dataframe_agent` function, which uses a DataFrame containing movie data from IMDb to answer user queries intelligently.

The code allows for queries like "Give me a good drama movie released recently" or other questions about movies using the IMDb dataset.

## Prerequisites
Before running the project, ensure the following libraries and services are installed and set up:
- Python 3.x
- Pip (for managing Python packages)

### Required Libraries:
1. `langchain`: This is the main framework used to build the agent.
2. `langchain_openai`: This is used to interface with OpenAI's API for GPT-4.
3. `pandas`: For working with the movie dataset.
4. `kagglehub`: For downloading datasets from Kaggle.
5. `getpass`: For securely entering your OpenAI API key.

You can install the necessary libraries with the following commands:

```bash
pip install langchain
pip install langchain_openai
pip install kagglehub
pip install pandas
pip install langchain_experimental
```

### Setup and Execution

1. Obtain the IMDb Dataset using kagglehub:

	```bash
	import kagglehub
	import os

	# Download latest version of the IMDb dataset
	path = kagglehub.dataset_download("fernandogarciah24/top-1000-imdb-dataset")
	file_path = os.path.join(path, 'imdb_top_1000.csv')
	df = pd.read_csv(file_path)
	```

2. Setup OpenAI API Key:
To use OpenAI’s GPT-4 API, you need to provide an API key. You can obtain this API key by signing up for OpenAI services. Once you have the API key, you can set it up in the code using the following snippet:

	```bash
	import getpass

	# Enter your OpenAI API Key (It will save it in the environment variable)
	if not os.environ.get("OPENAI_API_KEY"):
	    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
	```

3. Initialize the LLM (Language Model):
This section initializes the GPT-4 model using the provided OpenAI API key.

	```bash
	from langchain_openai import ChatOpenAI

	# Initialize GPT-4 model
	llm = ChatOpenAI(
	temperature=0,
	model="gpt-4-turbo",
	openai_api_key=os.environ["OPENAI_API_KEY"],
	streaming=True,
	)
	```

4. Create the Pandas DataFrame Agent:
The agent is created by passing the loaded DataFrame (df) to the create_pandas_dataframe_agent function. This allows the agent to process and respond to queries based on the data stored in the DataFrame. The agent can answer questions about movies in the dataset, such as filtering by genre, rating, or release year.

	```bash
	from langchain_experimental.agents import create_pandas_dataframe_agent

	# Create the agent
	agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True)
	```

5. Run the Agent with a Query:
You can now run the agent with a query to get movie recommendations or information. The agent will process the query and provide a response based on the data in the DataFrame.

	```bash
	example_query = "Please give me a good drama movie released recently."
	response = agent.run(example_query)
	print(response)
	```

6. Viewing Results:
Once the query is run, the agent will return a response based on the available movie data. You can change the query to suit different types of movie-related inquiries, such as filtering by genre, release year, rating, and more.

### create_pandas_dataframe_agent

The create_pandas_dataframe_agent function from Langchain is specifically designed to allow a large language model (LLM) like GPT-4 to directly interact with data stored in a pandas DataFrame. This function provides an easy way to query structured data, enabling the LLM to use the DataFrame as an external knowledge source while processing user queries.

#### Why use create_pandas_dataframe_agent?

The key reason for using create_pandas_dataframe_agent in this project, as opposed to a Retrieval-Augmented Generation (RAG) approach, is the direct and efficient interaction with the dataset stored in the DataFrame. Here’s why it’s preferred:
1. **Data-Specific Queries:**
The movie dataset is stored in a structured tabular format (CSV file) that allows the agent to directly search and filter the data. create_pandas_dataframe_agent is optimized for such structured data, providing a more efficient way to answer specific, data-driven queries.
2. **Simplicity and Directness:**
create_pandas_dataframe_agent simplifies the process of querying a DataFrame. It allows the model to directly access the DataFrame’s rows and columns, which makes it a straightforward choice for this project. There’s no need to complicate the interaction with additional indexing or search systems like those used in RAG.
3. **No Need for Document Retrieval:**
In RAG, the model retrieves relevant documents from a knowledge base before generating an answer. While this is great for scenarios involving unstructured data, in this case, the dataset is already structured in a DataFrame. The model can directly filter, query, and aggregate data from the DataFrame without needing an external retrieval step.

## Current State of the Agent

### What the Agent Does Well:
- **Efficient Query Handling:** The agent can process various types of queries about movies, including genres, release dates, and more.
- **Accurate Responses:** Based on the given IMDb dataset, the agent returns relevant responses.
- **Flexible and Adaptable:** The agent is capable of being adapted to other datasets, provided they are in a pandas DataFrame format.

### Limitations or Areas for Improvement:
- **Limited Context Understanding**: The agent’s responses are strictly tied to the dataset. If the query involves information outside the dataset (e.g., specific movie details not available in the top 1000 list), the agent may not be able to answer.
- **Performance on Complex Queries**: If the dataset grows, the agent may experience latency or performance issues, especially if the queries require complex operations.
- **Streaming Mode**: While the model supports streaming, the handling of large responses may need optimization.

### Suggestions for Future Enhancements:
- **Expand Dataset Coverage**: Integrating more data sources, such as movie reviews, cast information, and director details, could provide a more holistic response to user queries.
- **Scalability with RAG:**
As the dataset grows larger or if the agent needs to interact with unstructured data sources (like documents, websites, or text-based knowledge bases), implementing RAG can improve scalability. RAG integrates document retrieval into the response generation process, allowing the model to pull relevant information from large, external knowledge sources before generating an answer. This is particularly useful when dealing with more complex queries or when working with unstructured data that doesn’t fit neatly into a DataFrame. Potential benefits of RAG:
	- **Handling Larger Datasets:** RAG can pull relevant context from a large set of documents, enabling the agent to handle complex queries beyond what is available in the current DataFrame.
	- **Expanding the Knowledge Base:** By combining a retrieval mechanism with the model’s generation ability, RAG can make the agent more versatile, allowing it to tap into a broader range of information.
- **Improve Query Understanding**: Enhance the agent’s ability to understand more nuanced and complex queries, potentially using natural language processing techniques to process user inputs better.
- **Cache Results**: Implementing a caching mechanism could speed up responses for frequently asked queries, reducing computational overhead.
- **Add User Feedback**: Allow the agent to learn from user feedback (e.g., thumbs up/down) to improve its responses over time.

## Conclusion

This agent provides a foundation for building an intelligent movie recommendation system or answering queries related to a movie dataset. While the agent performs well with simple queries, there’s room for growth, especially with regards to expanding the dataset and enhancing the model’s capability to handle more complex queries. By leveraging Langchain, OpenAI, and structured data, this agent showcases the potential of combining AI and data to create intelligent agents for various applications.