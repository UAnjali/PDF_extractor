from chat.rag_chat import get_chain
import logging
import sys
from pathlib import Path

# Set up logging to file only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='chat.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

def start_chat():
    qa = None
    try:
        logger.info("Initializing Sales Counsellor Bot...")
        print("\nSales Counsellor Bot is initializing...")
        
        qa = get_chain()
        
        print("\n✅ Sales Counsellor Bot is ready!")
        print("Starting a new session...")
        print("Type 'exit' to end the session or 'help' for available commands.")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() == 'exit':
                    print("Goodbye!")
                    break
                    
                if query.lower() == 'help':
                    print("Commands: exit, help")
                    continue
                
                # Get response from the chain
                response = qa.invoke({"question": query})
                
                # Print only the answer from the response
                if isinstance(response, dict) and 'answer' in response:
                    print("\nBot:", response['answer'].strip())
                else:
                    print("\nBot:", str(response).strip())
                    
            except KeyboardInterrupt:
                print("\nSession interrupted. Archiving conversation...")
                break
            except Exception as e:
                logger.error(f"Error during chat: {str(e)}")
                print("Error occurred. Please try again.")
                continue
    finally:
        # Archive and summarize session if possible
        if qa and hasattr(qa, 'conversation_manager'):
            qa.conversation_manager.clear_current_session()
        print("Session archived and summarized (if possible).")

if __name__ == "__main__":
    # Add the project root to Python path if needed
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    start_chat()
