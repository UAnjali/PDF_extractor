import os
import yaml
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Any
# Import the summarizer
# from utils.conversation_summarizer import summarize_and_append_to_csv

# Set up logging
logging.basicConfig(
    filename='conversation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ConversationManager:
    def __init__(self):
        self.history_dir = Path("conversation_history")
        self.history_dir.mkdir(exist_ok=True)
        self.current_session_file = None
        self.session_start_time = None

    def _generate_session_filename(self) -> str:
        """Generate a unique filename for the session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"conversation_{timestamp}.yaml"

    def start_new_session(self):
        """Start a new conversation session."""
        self.session_start_time = datetime.now()
        self.current_session_file = self.history_dir / self._generate_session_filename()
        
        # Initialize the YAML file with session metadata
        initial_data = {
            "session_start": self.session_start_time.isoformat(),
            "conversation": []
        }
        
        try:
            with open(self.current_session_file, 'w') as f:
                yaml.dump(initial_data, f, default_flow_style=False)
            logging.info(f"Started new conversation session: {self.current_session_file}")
        except Exception as e:
            logging.error(f"Error creating new session file: {str(e)}")
            raise

    def add_message(self, role: str, content: str):
        """Add a message to the current session."""
        if not self.current_session_file:
            self.start_new_session()

        try:
            # Read existing conversation
            with open(self.current_session_file, 'r') as f:
                data = yaml.safe_load(f) or {"conversation": []}

            # Add new message
            message = {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content
            }
            data["conversation"].append(message)

            # Write back to file
            with open(self.current_session_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            logging.info(f"Added message to {self.current_session_file}")
        except Exception as e:
            logging.error(f"Error adding message to conversation: {str(e)}")
            raise

    def get_current_session_history(self) -> List[Dict[str, Any]]:
        """Get the current session's conversation history."""
        if not self.current_session_file or not self.current_session_file.exists():
            return []

        try:
            with open(self.current_session_file, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("conversation", [])
        except Exception as e:
            logging.error(f"Error reading conversation history: {str(e)}")
            return []

    def clear_current_session(self):
        """Clear the current session and start a new one."""
        if self.current_session_file and self.current_session_file.exists():
            try:
                # Archive the current session with a "completed" timestamp
                with open(self.current_session_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                data["session_end"] = datetime.now().isoformat()
                
                with open(self.current_session_file, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
                
                logging.info(f"Archived conversation session: {self.current_session_file}")
                # Summarize and append to CSV
                # csv_path = Path("conversation_summaries.csv")
                # summarize_and_append_to_csv(str(self.current_session_file), str(csv_path))
            except Exception as e:
                logging.error(f"Error archiving conversation session: {str(e)}")
        
        # Start a new session
        self.start_new_session() 