# Mental Health Chatbot API

This project implements a FastAPI server with two endpoints: `/rag` for Retrieval-Augmented Generation and `/classification` for text classification.

## Setup

1. Clone the repository:
    git clone https://github.com/yourusername/mental-health-chatbot.git
    cd mental-health-chatbot

2. Create a virtual environment and activate it:
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate

3. Install the required packages:
    pip install -r requirements.txt


## Running the Application

1. To run the application locally:
    uvicorn app.main:app --reload
    The server will be accessible at `http://localhost:8000`.

2. To run the application using Docker:
    docker build -t mental-health-chatbot .
    docker run -p 8000:8000 mental-health-chatbot
    
## API Endpoints

1. RAG Endpoint:
- URL: `/rag`
- Method: POST
- Input: JSON object with a `prompt` field
- Output: JSON object with `response` and `relevant_articles` fields

2. Classification Endpoint:
- URL: `/classification`
- Method: POST
- Input: JSON object with a `text` field
- Output: JSON object with a `classification` field

## Testing the API

You can use curl or any API testing tool like Postman to test the endpoints:

1. RAG Endpoint:
    curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"prompt":"I'm feeling anxious"}'

2. Classification Endpoint:
    curl -X POST "http://localhost:8000/classification" -H "Content-Type: application/json" -d '{"text":"I can't sleep at night"}'

## Note

This is a basic implementation and may need further refinement for production use. Ensure to use appropriate security measures and error handling in a production environment.

Setup and Execution Instructions:

To set up and run the application:

Ensure you have Python 3.9+ installed.
Clone the repository and navigate to the project directory.
Create a virtual environment and activate it.
Install the requirements using pip install -r requirements.txt.
Run the application using uvicorn app.main:app --reload.

For Docker deployment:

Ensure Docker is installed on your system.
Build the Docker image: docker build -t mental-health-chatbot .
Run the Docker container: docker run -p 8000:8000 mental-health-chatbot

The API will be accessible at http://localhost:8000. You can use the provided curl commands in the README to test the endpoints.