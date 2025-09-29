"""
Example usage of the chat shrinking pipeline with real data.
This script demonstrates how to use the shrink_chat function with the largeInput.json file.
"""

import json
import os
from chat_shrink import shrink_chat
from dotenv import load_dotenv
import time

load_dotenv()  # Load environment variables from .env file

def load_chat_from_json(file_path: str) -> list:
    """Load chat history from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the chat history from the data structure
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid JSON structure for chat history")


def main():
    """Main function to demonstrate chat shrinking with real data."""
    try:
        # Load the large input JSON
        chat_history = load_chat_from_json('largeInput.json')
        
        print(f"Original chat history: {len(chat_history)} messages")
        
        # Calculate approximate original token count
        original_content_length = sum(len(msg.get('content', '')) for msg in chat_history)
        estimated_original_tokens = original_content_length // 4  # Rough estimate
        
        print(f"Estimated original tokens: ~{estimated_original_tokens}")
        
        # Test with different model context window and output sizes
        model_configs = [
            {"model": "gemini-2.0-flash", "context_window": 1000000, "output_window": 8000, "name": "Gemini 2.0 Flash"},
            {"model": "gpt-4o", "context_window": 128000, "output_window": 4000, "name": "GPT-4o"},
            {"model": "gpt-4", "context_window": 32000, "output_window": 4000, "name": "GPT-4"},
        ]
        
        for config in model_configs:
            model_name = config["model"]
            max_tokens = config["context_window"]
            display_name = config["name"]
            
            print(f"\n--- Testing with {display_name} (Context: {max_tokens:,} tokens) ---")
            
            try:
                time.sleep(1)  # Brief pause to avoid rate limits

                start = time.time()
                output_tokens = config["output_window"]
                shrunk_chat = shrink_chat(chat_history, max_tokens=max_tokens, output_tokens=output_tokens, model_name=model_name)
                end = time.time()
                
                print(f"Shrinking took {end - start:.2f} seconds")
                print(f"Shrunk to {len(shrunk_chat)} messages")

                # Approximate token count after shrinking
                shrunk_content_length = sum(len(msg.get('content', '')) for msg in shrunk_chat)
                estimated_shrunk_tokens = shrunk_content_length // 4  # Rough estimate
                print(f"Estimated shrunk tokens: ~{estimated_shrunk_tokens}")
                
                reduction_percent = ((estimated_original_tokens - estimated_shrunk_tokens) / estimated_original_tokens * 100) if estimated_original_tokens > 0 else 0
                print(f"Token reduction: {reduction_percent:.1f}%")
                
                # Save shrunk version
                # Ensure shrunk_chats directory exists
                os.makedirs('shrunk_chats', exist_ok=True)
                
                # Save in shrunk_chats directory
                output_filename = f'shrunk_chats/shrunk_chat_{model_name.replace("-", "_")}_{max_tokens}.json'
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump({
                        'metadata': {
                            'original_messages': len(chat_history),
                            'shrunk_messages': len(shrunk_chat),
                            'model_context_window': max_tokens,
                            'model_output_window': output_tokens,
                            'model_name': model_name,
                            'processing_time_seconds': end - start,
                            'estimated_original_tokens': estimated_original_tokens,
                            'estimated_shrunk_tokens': estimated_shrunk_tokens,
                            'token_reduction_percentage': reduction_percent,
                            'timestamp': time.time()
                        },
                        'data': shrunk_chat
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"     💾 Saved to shrunk_chats folder: {os.path.basename(output_filename)}")
                
            except Exception as e:
                print(f"     Error: {str(e)}")
        
    except Exception as e:
        print(f"Failed to process chat history: {str(e)}")


if __name__ == "__main__":
    main()