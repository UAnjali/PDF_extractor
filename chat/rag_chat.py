import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from config.config import load_config

# Set up logging
logging.basicConfig(
    filename='chat.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_base_prompt() -> str:
    """Load the base prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "base_prompt.txt"
    with open(prompt_path, "r") as f:
        return f.read()

class SalesCounsellorBot:
    def __init__(self):
        self.config = load_config()
        self.initialize_chain()
        self.is_first_message = True
        # Define keyword mappings for different topics with more variations
        self.keyword_mappings = {
            "data analytics": ["data analysis", "analytics course", "data science", "business analytics", "data analyst", "analytics training"],
            "curriculum": ["course topics", "modules", "class topics", "syllabus", "course content", "learning path", "what will i learn", "topics covered"],
            "course topic": ["curriculum", "modules", "syllabus", "course content", "learning path", "what will i learn", "topics covered"],
            "topics": ["curriculum", "course content", "modules", "syllabus", "learning path", "what will i learn"],
            "projects": ["case studies", "hands on", "practice", "real world projects", "industry projects"],
            "placement": ["jobs", "career", "salary", "companies", "recruitment", "hiring", "job opportunities", "placement support"],
            "fee": ["payment", "pricing", "cost", "installment", "refund", "scholarship", "course fee", "payment options", "price", "costing"],
            "duration": ["course length", "time period", "schedule", "batch timing", "class schedule", "how long", "course duration"],
            "prerequisites": ["eligibility", "requirements", "background", "skills needed", "qualification", "who can join"]
        }
        
        # Define query understanding prompt
        self.query_understanding_prompt = """Analyze the following question and extract key information. Focus on understanding the intent even if the question is in informal language or different format.

        1. Specific question or aspects or query which is being asked about
        2. Any implicit requirements or constraints
        3. Context from previous conversation only if relevant
        4. Query asked by the learner (including informal language)
        
        Question: {question}
        Chat History: {chat_history}
        
        Provide a structured search query that will help find the most relevant information in the knowledge base.
        Important:
        - Understand informal language and variations (e.g., "bta do", "tell me", "what about")
        - Focus on the question being asked about
        - Include common variations of the topic
        - Consider the context from conversation history
        - Format the search query to be clear according to the knowledge base and the topics.
        
        Examples:
        Input: "course topic bta do yar"
        Output: "course topics curriculum modules syllabus what will i learn topics covered"
        
        Input: "Help with the DA Course"
        Output: "data analytics course overview curriculum modules syllabus what will i learn topics covered"
        
        Input: "Tell me about the course"
        Output: "data analytics course overview curriculum modules syllabus what will i learn topics covered"
        
        Format the search query to be clear and comprehensive."""

    def initialize_chain(self):
        try:
            # Initialize the LLM with Gemini 2.5 Flash Preview
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                google_api_key=self.config["GOOGLE_API_KEY"],
                temperature=1,
                convert_system_message_to_human=True,
                generation_config={
                    "max_output_tokens": 2048,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )

            # Initialize embeddings with OpenAI text-embedding-3-large
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=self.config["OPENAI_API_KEY"],
                dimensions=3072  # Using the larger dimension size for better performance
            )

            # Initialize Qdrant client
            qdrant_client = QdrantClient(
                url=self.config["QDRANT_URL"],
                api_key=self.config["QDRANT_API_KEY"]
            )

            # Check if collection exists, create if it doesn't
            collection_name = self.config["QDRANT_COLLECTION"]
            try:
                collection_info = qdrant_client.get_collection(collection_name)
                logging.info(f"Using existing collection '{collection_name}'")
            except Exception as e:
                logging.info(f"Creating new collection '{collection_name}'")
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "size": 3072,  # Size for text-embedding-3-large
                        "distance": "Cosine"
                    }
                )

            # Initialize vector store
            self.vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings
            )

            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Load base prompt
            base_prompt = load_base_prompt()

            # Define the prompt template using the base prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{base_prompt}\n\nContext: {{context}}\n\nChat History: {{chat_history}}\n\nQuestion: {{question}}\n\nIs this the first message in the conversation? {{is_first_message}}"),
                ("human", "{question}")
            ])

            # Create the chain
            self.chain = prompt | llm

            logging.info("Chat chain initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing chat chain: {str(e)}")
            raise

    def is_basic_conversation(self, question: str) -> bool:
        """Check if the question is basic conversation or greeting."""
        basic_phrases = [
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
            "good evening", "how are you", "how's it going", "what's up",
            "thanks", "thank you", "bye", "goodbye", "see you"
        ]
        question_lower = question.lower().strip()
        return any(phrase in question_lower for phrase in basic_phrases)

    def enhance_query(self, question: str) -> str:
        """Enhance the search query with related keywords."""
        question_lower = question.lower()
        enhanced_query = question
        
        # Always add data analytics related keywords for general queries
        if len(question.split()) <= 10:  # For short/general queries
            enhanced_query += " data analytics course"
        
        # Add relevant keywords based on the question
        for topic, keywords in self.keyword_mappings.items():
            # Check for topic and its variations in the question
            if (topic in question_lower or 
                any(keyword in question_lower for keyword in keywords[:2])):  # Check first two keywords as well
                # Add 3-4 most relevant keywords to the query
                enhanced_query += " " + " ".join(keywords[:4])
                break
        
        logging.info(f"Original query: {question}")
        logging.info(f"Enhanced query: {enhanced_query}")
        return enhanced_query

    def understand_query(self, question: str, chat_history: List[tuple]) -> str:
        """Use LLM to understand and structure the query."""
        try:
            # Initialize a separate LLM for query understanding
            query_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                google_api_key=self.config["GOOGLE_API_KEY"],
                temperature=0.1,  # Lower temperature for more focused results
                convert_system_message_to_human=True,
                generation_config={
                    "max_output_tokens": 1024,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            
            # Create prompt for query understanding
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.query_understanding_prompt),
                ("human", "{question}")
            ])
            
            # Get structured query
            chain = prompt | query_llm
            result = chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            
            structured_query = result.content if hasattr(result, 'content') else str(result)
            logging.info(f"Structured query: {structured_query}")
            return structured_query
            
        except Exception as e:
            logging.error(f"Error in query understanding: {str(e)}")
            return question  # Fallback to original question

    def get_response(self, question: str, chat_history: List[Dict[str, Any]] = None) -> str:
        try:
            if chat_history is None:
                chat_history = []

            # Convert chat history to the format expected by the chain
            formatted_history = []
            for msg in chat_history:
                if msg["role"] == "user":
                    formatted_history.append(("human", msg["content"]))
                else:
                    formatted_history.append(("ai", msg["content"]))

            # For basic conversation, don't require context
            if self.is_basic_conversation(question):
                logging.info("Handling basic conversation query")
                chain_input = {
                    "context": "This is a basic conversation query.",
                    "chat_history": formatted_history,
                    "question": question,
                    "is_first_message": str(self.is_first_message)
                }
            else:
                # First, understand and structure the query
                structured_query = self.understand_query(question, formatted_history)
                
                # Then enhance with keywords
                enhanced_query = self.enhance_query(structured_query)
                
                try:
                    # Search for relevant documents with logging
                    logging.info(f"Searching with enhanced structured query: {enhanced_query}")
                    docs = self.vector_store.similarity_search(
                        enhanced_query, 
                        k=3
                    )
                    
                    if not docs:
                        logging.info("No relevant documents found in knowledge base")
                        return "I want to ensure you get the most accurate and detailed information about this. Since this specific detail is not in my knowledge base, I'd be happy to schedule a call with one of our expert counsellors who can provide you with comprehensive information. Would you like me to help you schedule a call?"
                    
                    # Log the retrieved documents with relevance scores
                    for i, doc in enumerate(docs):
                        logging.info(f"Retrieved document {i+1}:")
                        logging.info(f"Content: {doc.page_content[:200]}...")
                        if hasattr(doc.metadata, 'source'):
                            logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        if hasattr(doc.metadata, 'course_type'):
                            logging.info(f"Course Type: {doc.metadata.get('course_type', 'Unknown')}")
                        if hasattr(doc, 'score'):
                            logging.info(f"Relevance Score: {doc.score}")

                    # Combine context with relevance information
                    context_parts = []
                    for doc in docs:
                        relevance = f"[Relevance: {doc.score:.2f}]" if hasattr(doc, 'score') else ""
                        context_parts.append(f"{doc.page_content} {relevance}")
                    
                    context = "\n\n".join(context_parts)
                    logging.info(f"Combined context length: {len(context)} characters")

                except Exception as search_error:
                    logging.error(f"Error during document search: {str(search_error)}")
                    return "I want to ensure you get the most accurate and detailed information about this. Since I encountered an issue accessing the information, I'd be happy to schedule a call with one of our expert counsellors who can provide you with comprehensive information. Would you like me to help you schedule a call?"

                # Prepare the input for the chain
                chain_input = {
                    "context": context,
                    "chat_history": formatted_history,
                    "question": question,  # Use original question for response generation
                    "is_first_message": str(self.is_first_message)
                }

            # Get response from the chain
            response = self.chain.invoke(chain_input)
            
            # Update first message flag
            self.is_first_message = False
            
            if hasattr(response, 'content'):
                logging.info(f"Generated response: {response.content[:200]}...")  # Log first 200 chars
                return response.content
            else:
                logging.warning(f"Unexpected response type: {type(response)}")
                return str(response)

        except Exception as e:
            logging.error(f"Error getting response: {str(e)}")
            # Return the counsellor call message as per base prompt
            return "I want to ensure you get the most accurate and detailed information about this. Since I encountered an error while processing your request, I'd be happy to schedule a call with one of our expert counsellors who can provide you with comprehensive information. Would you like me to help you schedule a call?"

    def clear_history(self):
        """Clear the conversation history and reset first message flag"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.is_first_message = True  # Reset first message flag
