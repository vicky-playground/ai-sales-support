from llama_index import SimpleDirectoryReader
# load the document
documents = SimpleDirectoryReader(input_files=["catalog.txt"]).load_data()


# Import necessary libraries
import torch
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, LangchainEmbedding
# Llamaindex also works with langchain framework to implement embeddings 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.prompts.prompts import SimpleInputPrompt
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
llm_hub = None
embeddings = None

# Replace 'Watsonx_API' and 'Project_id' with your actual API key and Project ID
Watsonx_API = "uvnQIfnjPk2Jpszy0hAvr80xCUAudclZsltCi3gYxAVu"
Project_id= "177ab670-c7d0-4f34-894f-228297d644d9"
    

# Function to initialize the Watsonx language model and its embeddings used to represent text data in a form (vectors) that machines can understand. 
def init_llm():
    global llm_hub, embeddings
    
    params = {
        GenParams.MAX_NEW_TOKENS: 250, # The maximum number of tokens that the model can generate in a single run.
        GenParams.MIN_NEW_TOKENS: 1,   # The minimum number of tokens that the model should generate in a single run.
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE, # The method used by the model for decoding/generating new tokens. In this case, it uses the sampling method.
        GenParams.TEMPERATURE: 0.8,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
        GenParams.TOP_K: 50,          # The top K parameter restricts the token generation to the K most likely tokens at each step, which can help to focus the generation and avoid irrelevant tokens.
        GenParams.TOP_P: 1            # The top P parameter, also known as nucleus sampling, restricts the token generation to a subset of tokens that have a cumulative probability of at most P, helping to balance between diversity and quality of the generated text.
    }
    
    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : Watsonx_API
    }

    
    model = Model(
        model_id= 'meta-llama/llama-2-70b-chat',
        credentials=credentials,
        params=params,
        project_id=Project_id)

    llm_hub = WatsonxLLM(model=model)

    # Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )

init_llm()

# Load the file
documents = SimpleDirectoryReader(input_files=["catalog.txt"]).load_data()

# LLMPredictor: to generate the text response (Completion)
llm_predictor = LLMPredictor(
        llm=llm_hub
)
                                 
# Hugging Face models can be supported by using LangchainEmbedding to convert text to embedding vector	
embed_model = LangchainEmbedding(embeddings)

# ServiceContext: to encapsulate the resources used to create indexes and run queries    
service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, 
        embed_model=embed_model
)    

# Build index
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# use a query engine as the interface for your question
query_engine = index.as_query_engine(service_context=service_context)

# Store the conversation history in a List
conversation_history = []

def ask_bot(input_text):

    PROMPT_QUESTION = """  
        You are a sales assistant working for IBM. You are dedicated to every client's success. You don't have a name and you don't need to mention it if you are not asked to answer your name.
        You are an expert in IBM products and helping a client to find the product they need.
        Your conversation with the human is recorded in the chat history below.

        History:
        "{history}"

        Now continue the conversation with the human without "```" and any inline code formatting. If you do not know the answer based on the chat history and the new input from the client, politely admit it and therefore you need more information. 
        Human: {input}
        Assistant:"""
        
    # update conversation history
    global conversation_history
    history_string = "\n".join(conversation_history)
    print(f"history_string: {history_string}")
    
    # query LlamaIndex and the LLM for the AI's response
    output = query_engine.query(PROMPT_QUESTION.format(history=history_string, input=input_text))
    print(f"output: {output}")
    
    # update conversation history with user input and AI's response
    conversation_history.append(input_text)
    conversation_history.append(output.response)
    
    return output.response

import gradio as gr 

with gr.Blocks() as demo:
    gr.Markdown('# IBM Sales Assistant')
    gr.Markdown('## Your assistant to guide you to the right product.')
    gr.Markdown('### Sample messages:')
    gr.Markdown('#### :) who are you?')
    gr.Markdown('#### :) I want to deploy an app for free')
    gr.Markdown('#### :) what products do you have?')
    gr.Markdown('#### :) what is wastonx?')
    gr.Markdown('#### many more......')
    
    # create an input textbox and a submit button
    inputs=gr.Textbox(lines=4, label="Input Box", placeholder="Enter your text here")
    submit_btn = gr.Button("Submit") 
    # define the behavior of the submit button using the ask_bot function
    submit_btn.click(fn=ask_bot, inputs=[inputs], outputs=gr.Textbox(lines=4, label="Output Box") )

# launch the Gradio interface
# As a computer networking practice, we set the server name inside the launch function to be 0.0.0.0 so that our app can be accessible by anybody on your local network.
demo.launch(server_name="0.0.0.0")
