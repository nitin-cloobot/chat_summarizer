import os
import requests
from typing import List, Dict


def generate_response(content: List[Dict[str, str]], system_prompt: str, model: str) -> str:
    """
    Sends a chat message to Gemini 2.0 Flash API and returns the AI response text.
    
    Args:
        content: List of message dictionaries with 'role' and 'content' keys
        system_prompt: System prompt for the LLM
        model: Model name (ignored for Gemini, always uses gemini-2.0-flash)
        
    Returns:
        AI response text
    """
    
    api_key = os.getenv('LLM_API_KEY')
    if not api_key:
        return "[Gemini API Error]: LLM_API_KEY environment variable not set"
    
    endpoint = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': api_key
    }
    
    # Build the contents list from the input messages
    contents = []
    
    # Convert the input format to Gemini's expected format
    for message in content:
        role = message.get('role', 'user')
        message_content = message.get('content', '')
        
        # Gemini uses 'user' and 'model' roles, map accordingly
        if role == 'assistant':
            gemini_role = 'model'
        else:
            gemini_role = 'user'
            
        contents.append({
            "role": gemini_role,
            "parts": [{"text": message_content}]
        })
    
    # Build the payload
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": contents
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract the AI response
        ai_text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        
        return ai_text if ai_text else "[Gemini API Error]: Empty response received"
        
    except requests.exceptions.RequestException as e:
        return f"[Gemini API Error]: Request failed - {str(e)}"
    except KeyError as e:
        return f"[Gemini API Error]: Unexpected response format - {str(e)}"
    except Exception as e:
        return f"[Gemini API Error]: {str(e)}"
