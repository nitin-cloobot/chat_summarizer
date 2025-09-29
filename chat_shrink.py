"""
Conversation-shrinking pipeline for trimming oversized role-based JSON chat lists.

This module implements an automated pipeline that trims chat history to fit within
a target token limit while preserving system messages, the newest user message,
and maintaining technical fidelity in assistant responses.
"""

import tiktoken
from typing import List, Dict, Any
import re
import gemini_api as llm_api


def shrink_chat(chat_history: List[Dict[str, str]], max_tokens: int, output_tokens: int, model_name: str) -> List[Dict[str, str]]:
    """
    Return an optimised chat list ≤ max_tokens while respecting all constraints.
    
    Preserves conversation flow order:
    - If input ends with user message, output ends with user message
    - If input ends with assistant message, output ends with assistant message
    
    Args:
        chat_history: List of chat messages with 'role' and 'content' keys
        max_tokens: Model's context window size (total tokens allowed)
        output_tokens: Maximum tokens the model can generate in one response
        model_name: Model name to use for token counting and summarization
        
    Returns:
        Optimized chat history that fits within token limit and preserves conversation flow
        
    Raises:
        ValueError: If no system message is found or other validation errors
    """
    if not chat_history:
        raise ValueError("Chat history cannot be empty")
    
    # Step 1: Copy the first system message exactly; raise ValueError if missing
    system_messages = [msg for msg in chat_history if msg.get("role") == "system"]
    if not system_messages:
        raise ValueError("No system message found in chat history")
    
    system_message = system_messages[0]
    
    # Step 2: Keep only the last user message
    user_messages = [msg for msg in chat_history if msg.get("role") == "user"]
    last_user_message = user_messages[-1] if user_messages else None
    
    # Step 3: Append all assistant messages (order preserved)
    assistant_messages = [msg for msg in chat_history if msg.get("role") == "assistant"]
    
    # Determine conversation flow order based on the last message
    last_message = chat_history[-1] if chat_history else None
    conversation_ends_with_user = last_message and last_message.get("role") == "user"
    
    # Build provisional list maintaining proper conversation flow
    provisional_list = [system_message]
    
    if conversation_ends_with_user:
        # Scenario 1: Input ends with user message -> user message should be last
        provisional_list.extend(assistant_messages)
        if last_user_message:
            provisional_list.append(last_user_message)
    else:
        # Scenario 2: Input ends with assistant message -> assistant message should be last
        if last_user_message:
            provisional_list.append(last_user_message)
        provisional_list.extend(assistant_messages)
    
    # Step 4: Compute token count of the provisional list
    try:
        token_count = _count_tokens(provisional_list, model_name)
    except Exception as e:
        raise ValueError(f"Failed to count tokens: {str(e)}")
    
    # Step 5: If over the model's context window, start summarizing
    if token_count <= max_tokens:
        return provisional_list
    
    # Step 5.1-5.2: Summarize assistant messages first
    if assistant_messages:
        try:
            summarized_assistant = _summarize_assistant_messages(assistant_messages, model_name, max_tokens, output_tokens)
            
            # Rebuild provisional list maintaining conversation flow order
            provisional_list = [system_message]
            
            if conversation_ends_with_user:
                # User message should be last
                provisional_list.append(summarized_assistant)
                if last_user_message:
                    provisional_list.append(last_user_message)
            else:
                # Assistant message should be last
                if last_user_message:
                    provisional_list.append(last_user_message)
                provisional_list.append(summarized_assistant)
            
            token_count = _count_tokens(provisional_list, model_name)
        except Exception as e:
            raise ValueError(f"Failed to summarize assistant messages: {str(e)}")
    
    # Step 5.3: If still over budget, summarize the user message
    if token_count > max_tokens and last_user_message:
        try:
            summarized_user = _summarize_user_message(last_user_message, model_name, max_tokens, output_tokens)
            
            # Rebuild with summarized user message maintaining conversation flow
            provisional_list = [system_message]
            
            if conversation_ends_with_user:
                # User message should be last
                if assistant_messages:
                    provisional_list.append(summarized_assistant)
                provisional_list.append(summarized_user)
            else:
                # Assistant message should be last
                provisional_list.append(summarized_user)
                if assistant_messages:
                    provisional_list.append(summarized_assistant)
            
            token_count = _count_tokens(provisional_list, model_name)
        except Exception as e:
            raise ValueError(f"Failed to summarize user message: {str(e)}")
    
    # Step 6: Return the final list
    return provisional_list


def _count_tokens(messages: List[Dict[str, str]], model_name: str) -> int:
    """
    Count tokens in a list of chat messages using tiktoken.
    If model_name is unknown, defaults to "cl100k_base" encoding.
    
    Args:
        messages: List of chat messages
        model_name: Model name for token encoding
        
    Returns:
        Total token count
    """
    try:
        # Map common model names to tiktoken encodings
        encoding_map = {
            "gpt-4o": "cl100k_base",
            "gpt-4": "cl100k_base", 
            "gpt-3.5-turbo": "cl100k_base",
            "text-davinci-003": "p50k_base"
        }
        
        encoding_name = encoding_map.get(model_name, "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "")
            
            # Count tokens for role and content
            total_tokens += len(encoding.encode(role))
            total_tokens += len(encoding.encode(content))
            
            # Add overhead for message formatting (approximate)
            total_tokens += 4  # Overhead per message
        
        return total_tokens
    except Exception as e:
        raise Exception(f"Token counting failed: {str(e)}")


def _summarize_assistant_messages(assistant_messages: List[Dict[str, str]], model_name: str, max_tokens: int, output_tokens: int) -> Dict[str, str]:
    """
    Summarize all assistant messages into a single high-fidelity summary.
    
    Args:
        assistant_messages: List of assistant messages to summarize
        model_name: Model name for summarization
        max_tokens: Model's context window size
        output_tokens: Maximum tokens the model can generate in one response
        
    Returns:
        Single assistant message with summarized content
    """
    if not assistant_messages:
        return {"role": "assistant", "content": ""}
    
    # Combine all assistant content while preserving protected sections
    combined_content = ""
    for i, msg in enumerate(assistant_messages):
        content = msg.get("content", "")
        if i > 0:
            combined_content += "\n\n---\n\n"
        combined_content += content
    
    # Extract and preserve protected sections (code blocks, JSON, tables)
    protected_sections = _extract_protected_sections(combined_content)
    
    # Use optimized chunking based on model's capabilities
    chunks = _chunk_text_optimized(combined_content, max_tokens, output_tokens)
    
    # Summarize chunks with optimized strategy
    try:
        summarized_blocks = _summarise_blocks_optimized(chunks, model_name, max_tokens, output_tokens)
        
        # Restore protected sections in summary
        final_summary = _restore_protected_sections(summarized_blocks, protected_sections)
        
        return {"role": "assistant", "content": final_summary}
    except Exception as e:
        raise Exception(f"Assistant message summarization failed: {str(e)}")


def _summarize_user_message(user_message: Dict[str, str], model_name: str, max_tokens: int, output_tokens: int) -> Dict[str, str]:
    """
    Summarize a user message while preserving technical fidelity.
    
    Args:
        user_message: User message to summarize
        model_name: Model name for summarization
        max_tokens: Model's context window size for chunking optimization
        
    Returns:
        Summarized user message
    """
    content = user_message.get("content", "")
    
    # If content is already short, return as-is
    try:
        token_count = _count_tokens([user_message], model_name)
        if token_count <= 500:  # Reasonable threshold
            return user_message
    except:
        pass
    
    # Use optimized chunking strategy based on max_tokens
    chunks = _chunk_text_optimized(content, max_tokens, output_tokens)
    
    try:
        summarized_content = _summarise_blocks_optimized(chunks, model_name, max_tokens, output_tokens)
        return {"role": "user", "content": summarized_content}
    except Exception as e:
        raise Exception(f"User message summarization failed: {str(e)}")


def _extract_protected_sections(text: str) -> List[Dict[str, Any]]:
    """
    Extract protected sections (code blocks, JSON, tables) from text.
    
    Args:
        text: Text to extract protected sections from
        
    Returns:
        List of protected sections with metadata
    """
    protected_sections = []
    
    # Pattern for fenced code blocks
    code_block_pattern = r'```[\w]*\n(.*?)\n```'
    for match in re.finditer(code_block_pattern, text, re.DOTALL):
        protected_sections.append({
            'type': 'code_block',
            'content': match.group(0),
            'start': match.start(),
            'end': match.end(),
            'placeholder': f'__PROTECTED_CODE_{len(protected_sections)}__'
        })
    
    # Pattern for JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(json_pattern, text):
        # Simple heuristic to identify JSON-like structures
        if '"' in match.group(0) and ':' in match.group(0):
            protected_sections.append({
                'type': 'json',
                'content': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'placeholder': f'__PROTECTED_JSON_{len(protected_sections)}__'
            })
    
    # Pattern for Markdown tables
    table_pattern = r'\|.*?\|(?:\n\|.*?\|)*'
    for match in re.finditer(table_pattern, text, re.MULTILINE):
        if '|' in match.group(0):
            protected_sections.append({
                'type': 'table',
                'content': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'placeholder': f'__PROTECTED_TABLE_{len(protected_sections)}__'
            })
    
    # Sort by position (reverse order for safe replacement)
    protected_sections.sort(key=lambda x: x['start'], reverse=True)
    
    return protected_sections


def _restore_protected_sections(text: str, protected_sections: List[Dict[str, Any]]) -> str:
    """
    Restore protected sections in summarized text.
    
    Args:
        text: Summarized text with placeholders
        protected_sections: List of protected sections to restore
        
    Returns:
        Text with protected sections restored
    """
    result = text
    for section in protected_sections:
        placeholder = section['placeholder']
        if placeholder in result:
            result = result.replace(placeholder, section['content'])
    
    return result



def _chunk_text_optimized(text: str, max_tokens: int, output_tokens: int) -> List[str]:
    """
    Split text into optimized chunks based on model's context window and output size.
    Maximizes content per API call while respecting model's generation limits.
    
    Args:
        text: Text to chunk
        max_tokens: Model's total context window size
        output_tokens: Maximum tokens the model can generate in one response
        
    Returns:
        List of optimized text chunks
    """
    if not text:
        return []
    
    # Calculate optimal chunk size based on model's capabilities
    system_overhead = 1000  # Reserve for system prompts and formatting
    max_tokens_per_chunk = min(
        max_tokens - output_tokens - system_overhead,  # Space for context
        output_tokens * 4  # Rule of thumb: input:output ratio of 4:1
    )
    
    # If text is small enough for single API call, return as-is
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        total_tokens = len(encoding.encode(text))
        if total_tokens <= max_tokens_per_chunk:
            return [text]
    except:
        # Fallback to character-based estimation if tiktoken fails
        if len(text) <= max_tokens_per_chunk * 4:
            return [text]
    
    # Use character-based chunking with optimized chunk size
    chars_per_chunk = max_tokens_per_chunk * 4  # Rough estimate: 1 token ≈ 4 characters
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chars_per_chunk
        
        # Try to find a natural break point (sentence, paragraph, etc.)
        if end < len(text):
            # Look for sentence endings within the last 500 characters (larger window for bigger chunks)
            search_start = max(start, end - 500)
            sentence_break = text.rfind('.', search_start, end)
            if sentence_break > start:
                end = sentence_break + 1
            else:
                # Look for paragraph breaks
                para_break = text.rfind('\n\n', search_start, end)
                if para_break > start:
                    end = para_break + 2
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end
    
    return chunks


def _chunk_text(text: str, tokens_per_chunk: int = 2000) -> List[str]:
    """
    Legacy chunking function - kept for backward compatibility.
    Prefer _chunk_text_optimized for better performance.
    
    Args:
        text: Text to chunk
        tokens_per_chunk: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Simple chunking by character count (rough token estimate: 1 token ≈ 4 characters)
    chars_per_chunk = tokens_per_chunk * 4
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chars_per_chunk
        
        # Try to find a natural break point (sentence, paragraph, etc.)
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            search_start = max(start, end - 200)
            sentence_break = text.rfind('.', search_start, end)
            if sentence_break > start:
                end = sentence_break + 1
            else:
                # Look for paragraph breaks
                para_break = text.rfind('\n\n', search_start, end)
                if para_break > start:
                    end = para_break + 2
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end
    
    return chunks


def _summarise_blocks_optimized(blocks: List[str], model_name: str, max_tokens: int, output_tokens: int) -> str:
    """
    Optimized summarization that minimizes API calls by using model's context window efficiently.
    
    Args:
        blocks: List of text blocks to summarize
        model_name: Model name for summarization
        max_tokens: Model's context window size
        output_tokens: Maximum tokens the model can generate in one response
        
    Returns:
        Summarized text
    """
    if not blocks:
        return ""
    
    if len(blocks) == 1:
        # Single block - check if it needs summarization
        try:
            token_count = len(tiktoken.get_encoding("cl100k_base").encode(blocks[0]))
            if token_count <= 1000:
                return blocks[0]
        except:
            pass
    
    # Calculate optimal batch size based on model's capabilities
    system_overhead = 1000  # Reserve for system prompts and formatting
    max_input_tokens = max_tokens - output_tokens - system_overhead
    batch_size = max_input_tokens // (output_tokens * 1.5)  # Allow buffer for summarization
    
    system_prompt = "You are a professional high-fidelity condensing protocol agent."
    instruction = "Summarize the following text while preserving all technical details, data, and semantic meaning. Maintain code blocks, JSON structures, and tables exactly as they appear:\n\n"
    
    summarized_results = []
    current_batch = []
    current_batch_tokens = 0
    
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        instruction_tokens = len(encoding.encode(instruction))
        
        for block in blocks:
            block_tokens = len(encoding.encode(block))
            
            # Check if adding this block would exceed context window
            if current_batch_tokens + block_tokens + instruction_tokens > max_input_tokens and current_batch:
                # Process current batch
                batch_content = "\n\n---\n\n".join(current_batch)
                try:
                    content = [{
                        "role": "user",
                        "content": instruction + batch_content
                    }]
                    
                    response = llm_api.generate_response(
                        content=content,
                        system_prompt=system_prompt,
                        model=model_name
                    )
                    
                    summarized_results.append(response)
                    
                except Exception as e:
                    # If summarization fails, include original content
                    summarized_results.append(f"[Batch summarization failed: {str(e)}]\n{batch_content}")
                
                # Start new batch
                current_batch = [block]
                current_batch_tokens = block_tokens
            else:
                # Add to current batch
                current_batch.append(block)
                current_batch_tokens += block_tokens
    
    except Exception:
        # Fallback to character-based estimation if tiktoken fails
        max_chars = max_input_tokens * 4
        current_batch_chars = 0
        
        for block in blocks:
            block_chars = len(block)
            
            if current_batch_chars + block_chars > max_chars and current_batch:
                # Process current batch
                batch_content = "\n\n---\n\n".join(current_batch)
                try:
                    content = [{
                        "role": "user",
                        "content": instruction + batch_content
                    }]
                    
                    response = llm_api.generate_response(
                        content=content,
                        system_prompt=system_prompt,
                        model=model_name
                    )
                    
                    summarized_results.append(response)
                    
                except Exception as e:
                    summarized_results.append(f"[Batch summarization failed: {str(e)}]\n{batch_content}")
                
                # Start new batch
                current_batch = [block]
                current_batch_chars = block_chars
            else:
                current_batch.append(block)
                current_batch_chars += block_chars
    
    # Process final batch if exists
    if current_batch:
        batch_content = "\n\n---\n\n".join(current_batch)
        try:
            content = [{
                "role": "user",
                "content": instruction + batch_content
            }]
            
            response = llm_api.generate_response(
                content=content,
                system_prompt=system_prompt,
                model=model_name
            )
            
            summarized_results.append(response)
            
        except Exception as e:
            summarized_results.append(f"[Final batch summarization failed: {str(e)}]\n{batch_content}")
    
    # Combine all summarized results
    final_result = "\n\n".join(summarized_results)
    
    return final_result


def _summarise_blocks(blocks: List[str], model_name: str) -> str:
    """
    Legacy summarization function - kept for backward compatibility.
    Prefer _summarise_blocks_optimized for better performance.
    
    Args:
        blocks: List of text blocks to summarize
        model_name: Model name for summarization
        
    Returns:
        Summarized text
    """
    if not blocks:
        return ""
    
    if len(blocks) == 1:
        # Single block - check if it needs summarization
        try:
            token_count = len(tiktoken.get_encoding("cl100k_base").encode(blocks[0]))
            if token_count <= 1000:
                return blocks[0]
        except:
            pass
    
    system_prompt = "You are a professional high-fidelity condensing protocol agent."
    
    summarized_blocks = []
    
    for block in blocks:
        try:
            # Prepare the content for summarization
            content = [{
                "role": "user",
                "content": f"Summarize the following text while preserving all technical details, data, and semantic meaning. Maintain code blocks, JSON structures, and tables exactly as they appear:\n\n{block}"
            }]
            
            # Call LLM API for summarization
            response = llm_api.generate_response(
                content=content,
                system_prompt=system_prompt,
                model=model_name
            )
            
            summarized_blocks.append(response)
            
        except Exception as e:
            # If summarization fails, include original block but log the error
            summarized_blocks.append(f"[Summarization failed: {str(e)}]\n{block}")
    
    # Combine summarized blocks
    combined_summary = "\n\n".join(summarized_blocks)
    
    # Check if we need recursive summarization
    try:
        token_count = len(tiktoken.get_encoding("cl100k_base").encode(combined_summary))
        if token_count > 1000 and len(summarized_blocks) > 1:
            # Recursively summarize if still too large and multiple blocks
            return _summarise_blocks([combined_summary], model_name)
    except:
        pass
    
    return combined_summary


# Test harness
if __name__ == "__main__":
    import time
    
    # Test both scenarios for conversation flow with performance metrics
    
    # Scenario 1: Chat ending with user message
    test_chat_ending_with_user = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides technical guidance."
        },
        {
            "role": "user",
            "content": "Can you help me understand Python decorators?"
        },
        {
            "role": "assistant",
            "content": "Certainly! Python decorators are a powerful feature that allows you to modify or extend the behavior of functions or classes without permanently modifying their code. Here's a detailed explanation:\n\n```python\ndef my_decorator(func):\n    def wrapper(*args, **kwargs):\n        print(f\"Calling {func.__name__}\")\n        result = func(*args, **kwargs)\n        print(f\"Finished {func.__name__}\")\n        return result\n    return wrapper\n\n@my_decorator\ndef greet(name):\n    return f\"Hello, {name}!\"\n```\n\nThis example shows how decorators work by wrapping functions."
        },
        {
            "role": "user",
            "content": "Now I need to understand how to create a token counting system for a chat application that can handle very large conversations and needs to trim them intelligently while preserving important context."
        }
    ]
    
    # Scenario 2: Chat ending with assistant message
    test_chat_ending_with_assistant = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that provides technical guidance."
        },
        {
            "role": "user",
            "content": "Can you help me understand Python decorators?"
        },
        {
            "role": "assistant",
            "content": "Certainly! Python decorators are a powerful feature that allows you to modify or extend the behavior of functions or classes without permanently modifying their code."
        },
        {
            "role": "user",
            "content": "That's helpful, but can you show me more advanced examples?"
        },
        {
            "role": "assistant",
            "content": "Absolutely! Here are some advanced decorator patterns:\n\n1. **Parameterized Decorators:**\n\n```python\ndef retry(max_attempts=3):\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            for attempt in range(max_attempts):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_attempts - 1:\n                        raise e\n                    print(f\"Attempt {attempt + 1} failed: {e}\")\n        return wrapper\n    return decorator\n```\n\n2. **Class-based Decorators:**\n\n```python\nclass CountCalls:\n    def __init__(self, func):\n        self.func = func\n        self.count = 0\n    \n    def __call__(self, *args, **kwargs):\n        self.count += 1\n        print(f\"Call {self.count} of {self.func.__name__}\")\n        return self.func(*args, **kwargs)\n```"
        }
    ]
    
    try:
        print("=== PERFORMANCE OPTIMIZED CHAT SHRINKING DEMO ===")
        print("+ Using model context window for optimal chunking")
        print("+ Minimizing API calls by batching content efficiently")
        print("+ Preserving conversation flow order")
        print()
        
        print("=== Testing Scenario 1: Chat ending with USER message ===")
        print("Original chat history length:", len(test_chat_ending_with_user))
        print("Last message role:", test_chat_ending_with_user[-1]['role'])
        
        start_time = time.time()
        # Use GPT-4o's context window size (128k tokens)
        shrunk_chat_1 = shrink_chat(test_chat_ending_with_user, max_tokens=128000, model_name="gpt-4o")
        end_time = time.time()
        
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print("Shrunk chat history length:", len(shrunk_chat_1))
        print("Final message structure:")
        for i, message in enumerate(shrunk_chat_1):
            print(f"  {i+1}. Role: {message['role']}")
        print("Last message role in output:", shrunk_chat_1[-1]['role'])
        
        print()
        print("=== Testing Scenario 2: Chat ending with ASSISTANT message ===")
        print("Original chat history length:", len(test_chat_ending_with_assistant))
        print("Last message role:", test_chat_ending_with_assistant[-1]['role'])
        
        start_time = time.time()
        # Use GPT-4o's context window size (128k tokens)
        shrunk_chat_2 = shrink_chat(test_chat_ending_with_assistant, max_tokens=128000, model_name="gpt-4o")
        end_time = time.time()
        
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print("Shrunk chat history length:", len(shrunk_chat_2))
        print("Final message structure:")
        for i, message in enumerate(shrunk_chat_2):
            print(f"  {i+1}. Role: {message['role']}")
        print("Last message role in output:", shrunk_chat_2[-1]['role'])
        
        print()
        print("+ Both scenarios tested successfully with performance optimization!")
        print()
        print("PERFORMANCE BENEFITS:")
        print("  - Fewer API calls = faster processing")
        print("  - Better context utilization = higher quality summaries")
        print("  - Reduced costs = fewer API requests")
        print("  - Model-aware chunking = optimal token usage")
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        print("This is expected if llm_api module is not available in the test environment.")