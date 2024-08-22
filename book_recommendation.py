# Book recommendation created using langchain components in python3.11.7

import os
import logging
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community_.storage.ignite import GridGainStore
from langchain_community_.document_loaders.ignite import IgniteDocumentLoader
from langchain_community_.chat_message_histories.ignite import IgniteChatMessageHistory
from langchain_community.cache import InMemoryCache
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Initialize stores
    doc_loader = IgniteDocumentLoader(
        cache_name="review_cache",
        ignite_host="localhost",
        ignite_port=10800,
        create_cache_if_not_exists=True
    )
    logger.info("IgniteDocumentLoader initialized successfully.")

    key_value_store = GridGainStore(
        cache_name="laptop_specs",
        host="localhost",
        port=10800
    )
    logger.info("GridGainStore initialized successfully.")

    chat_history = IgniteChatMessageHistory(
        session_id="user_session",
        cache_name="chat_history",
        ignite_host="localhost",
        ignite_port=10800
    )
    logger.info("IgniteChatMessageHistory initialized successfully.")

except Exception as e:
    logger.error(f"Error initializing stores: {e}", exc_info=True)
    raise

# Initialize Gemini
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account-key.json"
from langchain_google_vertexai import VertexAI
llm = VertexAI(model_name="gemini-pro")
print(llm)

# Set up LLM cache
llm.cache = InMemoryCache()

# Create memory from IgniteChatMessageHistory
memory = ConversationBufferMemory(
    chat_memory=chat_history,
    return_messages=True,
    output_key="output"
)
print(memory)

# Define the conversation template
conversation_template = ChatPromptTemplate.from_messages([
    ("human", "You are a helpful AI assistant specializing in laptop recommendations. Use the provided laptop information to assist the user."),
    MessagesPlaceholder(variable_name="history"),
    ("human", """
Relevant Laptop Information:
{reviews}

Laptop Specifications:
{specs}

User Query: {input}
"""),
])

def format_conversation_history(history: List[HumanMessage | AIMessage]) -> List[HumanMessage | AIMessage]:
    """
    Format the conversation history.
    
    Args:
        history (List[HumanMessage | AIMessage]): The conversation history.
    
    Returns:
        List[HumanMessage | AIMessage]: The formatted conversation history.
    """
    return history  # We can return the history as is, since it's already in the correct format

def get_relevant_documents(query: str) -> List[Document]:
    """
    Retrieve relevant documents based on the user's query.
    
    Args:
        query (str): The user's query.
    
    Returns:
        List[Document]: A list of relevant documents.
    """
    try:
        logger.debug(f"Attempting to load documents for query: {query}")
        documents = doc_loader.load()
        logger.debug(f"Loaded {len(documents)} documents")
        if not documents:
            logger.warning("No documents were loaded from the cache")
            return []
        relevant_docs = documents[:2]  # Return top 2 documents for simplicity
        logger.debug(f"Returning {len(relevant_docs)} relevant documents")
        return relevant_docs
    except Exception as e:
        logger.error(f"Error in get_relevant_documents: {e}", exc_info=True)
        return []

def get_laptop_specs(query: str) -> Dict[str, str]:
    """
    Retrieve laptop specifications.
    
    Args:
        query (str): The user's query (not used in this function but kept for consistency).
    
    Returns:
        Dict[str, str]: A dictionary of laptop specifications.
    """
    try:
        specs = key_value_store.mget(["laptop1", "laptop2", "laptop3", "laptop4"])
        return dict(zip(["laptop1", "laptop2", "laptop3", "laptop4"], specs))
    except Exception as e:
        logger.error(f"Error in get_laptop_specs: {e}", exc_info=True)
        return {}

# Define the main chain for processing user input
main_chain = (
    RunnablePassthrough.assign(
        reviews=RunnableLambda(lambda _: '\n'.join([doc.page_content for doc in doc_loader.load()][:2])),
        specs=RunnableLambda(lambda _: key_value_store.mget(["laptop1", "laptop2", "laptop3", "laptop4"])),
        history=RunnableLambda(lambda _: chat_history.messages)
    )
    | conversation_template
    | llm
    | StrOutputParser()
)

def process_user_input(user_input: str) -> str:
    """
    Process the user's input and generate a response.
    
    Args:
        user_input (str): The user's input query.
    
    Returns:
        str: The generated response.
    """
    try:
        logger.debug(f"Processing user input: {user_input}")
        response = main_chain.invoke({"input": user_input})
        logger.debug(f"Generated response: {response}")
        
        # Update memory with the new interaction
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)
        
        return response
    except Exception as e:
        logger.error(f"Error in process_user_input: {e}", exc_info=True)
        return f"I apologize, but I encountered an error while processing your request. Please try again."

def populate_caches():
    """
    Populate caches with sample data for demonstration purposes.
    """
    try:
        # Populate review cache
        reviews = {
            "laptop1": "Great performance for coding and video editing. The 16GB RAM and dedicated GPU make multitasking a breeze.",
            "laptop2": "Excellent battery life, perfect for students. Lightweight and portable, but the processor is a bit slow for heavy tasks.",
            "laptop3": "High-resolution display, ideal for graphic design. Comes with a stylus, but the price is on the higher side.",
            "laptop4": "Budget-friendly option with decent specs. Good for everyday tasks, but struggles with gaming.",
        }
        doc_loader.populate_cache(reviews)
        
        # Populate specs cache
        specs = {
            "laptop1": "16GB RAM, NVIDIA RTX 3060, Intel i7 11th Gen",
            "laptop2": "8GB RAM, Intel Iris Xe Graphics, Intel i5 11th Gen",
            "laptop3": "32GB RAM, NVIDIA RTX 3080, AMD Ryzen 9",
            "laptop4": "8GB RAM, Intel UHD Graphics, Intel i3 10th Gen",
        }
        key_value_store.mset([(k, v) for k, v in specs.items()])

        # Verify cache contents
        logger.info("Verifying cache contents:")
        for key in reviews.keys():
            value = doc_loader.get(key)
            logger.info(f"Review cache entry for {key}: {value}")
        for key in specs.keys():
            value = key_value_store.mget([key])[0]
            logger.info(f"Specs cache entry for {key}: {value}")
        
        logger.info("Caches populated and verified with sample data.")
    except Exception as e:
        logger.error(f"Error populating caches: {e}", exc_info=True)
        raise  # Re-raise the exception to ensure it's not silently ignored

def main():
    """
    Main function to run the Laptop Recommendation System.
    """
    print("Welcome to the Laptop Recommendation System!")
    print("Populating caches with sample data...")
    populate_caches()
    
    print("You can ask questions about laptops or request recommendations.")
    print("Type 'exit' to end the conversation.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Thank you for using the Laptop Recommendation System. Goodbye!")
                break

            response = process_user_input(user_input)
            print(f"\nAI: {response}")
        except Exception as e:
            logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
            print("An error occurred. Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unhandled error occurred: {e}", exc_info=True)
        print("An unexpected error occurred. The program will now exit.")