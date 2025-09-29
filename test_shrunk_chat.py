"""
Test script to demonstrate shrunk chat functionality with Gemini API.
This script loads a shrunk chat, adds a test question, and generates a response using Gemini.
"""

import json
import os
import time
from typing import List, Dict
from dotenv import load_dotenv
import gemini_api
from chat_shrink import shrink_chat, _count_tokens

# Load environment variables
load_dotenv()


def load_shrunk_chat(filename: str) -> List[Dict[str, str]]:
    """Load a shrunk chat from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Invalid JSON structure")
    except FileNotFoundError:
        print(f"File {filename} not found. Please run example_usage.py first to generate shrunk chats.")
        return []


def test_shrunk_chat_with_gemini(chat_history: List[Dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
    """Test the shrunk chat directly with Gemini API without adding any questions.
    
    Returns:
        tuple: (response_text, updated_chat_history)
    """
    
    # Extract system prompt from the chat history
    system_prompt = ""
    for msg in chat_history:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
            break
    
    # If no system prompt found, use a default one
    if not system_prompt:
        system_prompt = "You are a helpful assistant. Answer based on the conversation context."
    
    print("🤖 Sending shrunk chat directly to Gemini API...")
    print(f"📊 Chat contains {len(chat_history)} messages")
    
    response = gemini_api.generate_response(
        content=chat_history,
        system_prompt=system_prompt,
        model="gemini-2.0-flash"
    )
    
    # Add the response to create the full conversation
    updated_chat = chat_history.copy()
    if not response.startswith("[Gemini API Error]"):
        updated_chat.append({
            "role": "assistant",
            "content": response
        })
    
    return response, updated_chat


def save_continued_chat(original_filename: str, continued_chat: List[Dict[str, str]], test_number: int):
    """Save the continued chat conversation to a new JSON file.
    
    Args:
        original_filename: Name of the original shrunk chat file
        continued_chat: The chat history with new responses
        test_number: Test question number for unique naming
    """
    
    # Ensure continued_chats directory exists
    os.makedirs('continued_chats', exist_ok=True)
    
    # Create output filename based on original, removing the directory part
    base_name = os.path.basename(original_filename).replace('.json', '')
    output_filename = f'continued_chats/{base_name}_continued_test_{test_number}.json'
    
    # Calculate some metadata
    original_messages = len([msg for msg in continued_chat if msg.get('role') != 'assistant' or 'test' not in msg.get('content', '').lower()])
    new_messages = len(continued_chat) - original_messages
    
    # Save the continued conversation
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Load original shrunk chat metadata
        original_metadata = {}
        try:
            with open(original_filename, 'r', encoding='utf-8') as orig_f:
                original_data = json.load(orig_f)
                original_metadata = original_data.get('metadata', {})
        except Exception:
            pass

        json.dump({
            'metadata': {
                'source_file': original_filename,
                'test_number': test_number,
                'total_messages': len(continued_chat),
                'original_messages': original_messages,
                'new_messages': new_messages,
                'original_metadata': original_metadata,  # Include all metadata from source file
                'model_name': 'gemini-2.0-flash',  # Model used for continuation
                'model_context_window': 1000000,
                'model_output_window': 8000,
                'timestamp': time.time(),
                'source_file_created': os.path.getctime(original_filename) if os.path.exists(original_filename) else None
            },
            'data': continued_chat
        }, f, indent=2, ensure_ascii=False)
    
    return output_filename


def analyze_chat_quality(original_file: str, shrunk_chat: List[Dict[str, str]]):
    """Analyze the quality and token reduction of the shrunk chat."""
    
    print("\n📊 CHAT ANALYSIS")
    print("=" * 50)
    
    # Load original for comparison if available
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        original_chat = original_data.get('data', []) if isinstance(original_data, dict) else original_data
        
        # Calculate token counts
        try:
            original_tokens = _count_tokens(original_chat, "gemini-2.0-flash")
            shrunk_tokens = _count_tokens(shrunk_chat, "gemini-2.0-flash")
            
            print(f"📈 Original chat: {len(original_chat)} messages, ~{original_tokens:,} tokens")
            print(f"📉 Shrunk chat: {len(shrunk_chat)} messages, ~{shrunk_tokens:,} tokens")
            print(f"💾 Token reduction: {((original_tokens - shrunk_tokens) / original_tokens * 100):.1f}%")
            
        except Exception as e:
            print(f"⚠️  Token counting failed: {e}")
            print(f"📈 Original chat: {len(original_chat)} messages")
            print(f"📉 Shrunk chat: {len(shrunk_chat)} messages")
            
    except FileNotFoundError:
        print("⚠️  Original chat file not found for comparison")
    
    # Show chat structure
    print(f"\n🏗️  SHRUNK CHAT STRUCTURE:")
    for i, msg in enumerate(shrunk_chat):
        role = msg.get('role', 'unknown')
        content_length = len(msg.get('content', ''))
        content_preview = msg.get('content', '')[:100] + "..." if content_length > 100 else msg.get('content', '')
        
        print(f"  {i+1}. {role.upper()}: {content_length} chars")
        print(f"     Preview: {content_preview}")
        print()


def main():
    """Main function to test shrunk chats with Gemini API."""
    
    print("🧪 SHRUNK CHAT TESTING WITH GEMINI API")
    print("=" * 50)
    
    # Check if API key is configured
    api_key = os.getenv('LLM_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("❌ Error: Please set your Gemini API key in .env file")
        print("   Copy .env.example to .env and add your key")
        return
    
    # Look for available shrunk chat files in shrunk_chats directory
    shrunk_files = [
        'shrunk_chats/shrunk_chat_gpt_4o_128000.json',
        'shrunk_chats/shrunk_chat_gpt_4_32000.json'
    ]
    
    available_files = [f for f in shrunk_files if os.path.exists(f)]
    
    if not available_files:
        print("❌ No shrunk chat files found in shrunk_chats directory!")
        print("   Please run 'python example_usage.py' first to generate shrunk chats")
        return
    
    # Test each available shrunk chat
    for filename in available_files:
        print(f"\n🔍 Testing: {filename}")
        print("-" * 40)
        
        # Load the shrunk chat
        shrunk_chat = load_shrunk_chat(filename)
        if not shrunk_chat:
            print("❌ Failed to load shrunk chat")
            continue
        
        # Analyze the chat quality
        analyze_chat_quality('largeInput.json', shrunk_chat)
        
        # Test the shrunk chat directly with Gemini API
        print(f"\n🎯 TESTING SHRUNK CHAT PERFORMANCE:")
        print("-" * 30)
        
        response, continued_chat = test_shrunk_chat_with_gemini(shrunk_chat)
        
        if response.startswith("[Gemini API Error]"):
            print(f"❌ API Error: {response}")
        else:
            print(f"✅ Gemini Response from Shrunk Chat:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
            # Save the continued chat
            try:
                output_file = save_continued_chat(filename, continued_chat, 1)
                print(f"\n💾 Saved continued chat to: {os.path.basename(output_file)}")
            except Exception as e:
                print(f"\n⚠️  Failed to save continued chat: {e}")
        
        # Ask user if they want to continue with next file
        if len(available_files) > 1 and filename != available_files[-1]:
            user_input = input(f"\n⏳ Continue testing next file? (y/n): ").lower().strip()
            if user_input != 'y':
                break
    
    print("\n✨ Testing completed!")
    print("\n💡 TIP: This test shows how the shrunk chat performs as a direct replacement")
    print("   for the original oversized chat, using fewer tokens for the same context!")
    print("\n📂 Check the 'continued_chats' folder for the generated responses")
    print("   These files show how the shrunk chats perform in actual conversations.")
    print("\n📁 Directory structure:")
    print("   - shrunk_chats/: Contains the shortened chat histories")
    print("   - continued_chats/: Contains the conversation continuations")


if __name__ == "__main__":
    main()