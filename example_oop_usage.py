"""
Example usage of the OOP Chat Summarizer.
This demonstrates how to use the refactored single-file OOP implementation.

INTEGRATION WITH YOUR EXISTING VENDOR HANDLERS:
===============================================

To use your existing vendor handlers, uncomment and modify the integration section below:
"""

import json
from chat_summarizer_oop import ChatSummarizerApp, ModelConfig, VendorFactory

# INTEGRATION SECTION - Uncomment and modify to use your existing handlers:
# from your_project.gemini_handler import YourGeminiHandler
# from your_project.openai_handler import YourOpenAIHandler
# 
# # Register your existing handlers
# VendorFactory.register_handler('gemini', YourGeminiHandler)
# VendorFactory.register_handler('openai', YourOpenAIHandler)

def load_chat_from_json(file_path: str) -> list:
    """Load chat history from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid JSON structure for chat history")

def main():
    """Main function to demonstrate OOP chat shrinking."""
    try:
        # Load the large input JSON
        chat_history = load_chat_from_json('largeInput.json')
        
        print(f"Original chat history: {len(chat_history)} messages")
        
        # Initialize the OOP app
        app = ChatSummarizerApp()
        
        # Test with different models
        model_configs = [
            {"model": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
            {"model": "gpt-4o", "name": "GPT-4o"},
            {"model": "gpt-4", "name": "GPT-4"},
        ]
        
        for config in model_configs:
            model_name = config["model"]
            display_name = config["name"]
            
            print(f"\n--- Testing with {display_name} ---")
            
            try:
                # Process the chat using OOP approach
                result = app.process_chat(chat_history, model_name)
                
                print(f"Processing time: {result['metadata']['processing_time_seconds']:.2f} seconds")
                print(f"Shrunk to {result['metadata']['shrunk_messages']} messages")
                print(f"Token reduction: {result['metadata']['token_reduction_percentage']:.1f}%")
                
                # Save result using OOP method
                output_filename = f'shrunk_chats/shrunk_chat_{model_name.replace("-", "_")}_{result["metadata"]["model_context_window"]}.json'
                app.save_result(result, output_filename)
                print(f"Saved to: {output_filename}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
        
        print("\nOOP Implementation Benefits:")
        print("  - Clean separation of concerns")
        print("  - Vendor abstraction with easy extension")
        print("  - Batch processing before API calls")
        print("  - Comprehensive error handling")
        print("  - Modular and testable design")
        print("\nTo integrate your existing vendor handlers:")
        print("  1. Import your handler classes")
        print("  2. Register them with VendorFactory.register_handler()")
        print("  3. The batching will work automatically with your handlers")
        
    except Exception as e:
        print(f"Failed to process chat history: {str(e)}")

if __name__ == "__main__":
    main()
