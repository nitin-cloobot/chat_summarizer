"""
Conversation-shrinking pipeline for trimming oversized role-based JSON chat lists.
Refactored using Object-Oriented Programming principles with vendor abstraction and batching.

This module implements an automated pipeline that trims chat history to fit within
a target token limit while preserving system messages, the newest user message,
and maintaining technical fidelity in assistant responses.

INTEGRATION WITH EXISTING VENDOR HANDLERS:
==========================================

To integrate your existing vendor handlers from your separate project:

1. Import your existing handler classes:
   ```python
   from your_project.gemini_handler import YourGeminiHandler
   from your_project.openai_handler import YourOpenAIHandler
   ```

2. Register them with the VendorFactory:
   ```python
   VendorFactory.register_handler('gemini', YourGeminiHandler)
   VendorFactory.register_handler('openai', YourOpenAIHandler)
   ```

3. Or replace the placeholder classes directly in the _handlers dict

The vendor handlers should implement the VendorHandler interface:
- process_request(model_config, messages) -> (success, BatchResult, error_message)
- Return format: (True, content, input_tokens, output_tokens, response.model)
"""

import tiktoken
import os
import requests
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    vendor: str
    context_window: int
    output_window: int
    api_key: Optional[str] = None


@dataclass
class BatchResult:
    """Result of a batch processing operation."""
    success: bool
    content: str
    input_tokens: int
    output_tokens: int
    error_message: Optional[str] = None


class VendorHandler(ABC):
    """Abstract base class for vendor-specific API handlers."""
    
    @abstractmethod
    def process_request(self, model_config: ModelConfig, messages: List[Dict[str, str]]) -> Tuple[bool, BatchResult, str]:
        """
        Process a request with the vendor's API.
        
        Args:
            model_config: Model configuration
            messages: List of messages to process
            
        Returns:
            Tuple of (success, BatchResult, error_message)
        """
        pass


class GeminiHandler(VendorHandler):
    """Handler for Google Gemini API - placeholder for external implementation."""
    
    def process_request(self, model_config: ModelConfig, messages: List[Dict[str, str]]) -> Tuple[bool, BatchResult, str]:
        """
        Process request using Gemini API.
        This is a placeholder - implement your own Gemini handler from your separate project.
        """
        # TODO: Import and use your existing Gemini handler implementation
        # Example:
        # from your_gemini_handler import YourGeminiHandler
        # handler = YourGeminiHandler()
        # return handler.process_request(model_config, messages)
        
        raise NotImplementedError("Please implement Gemini handler from your separate project")


class OpenAIHandler(VendorHandler):
    """Handler for OpenAI API - placeholder for external implementation."""
    
    def process_request(self, model_config: ModelConfig, messages: List[Dict[str, str]]) -> Tuple[bool, BatchResult, str]:
        """
        Process request using OpenAI API.
        This is a placeholder - implement your own OpenAI handler from your separate project.
        """
        # TODO: Import and use your existing OpenAI handler implementation
        # Example:
        # from your_openai_handler import YourOpenAIHandler
        # handler = YourOpenAIHandler()
        # return handler.process_request(model_config, messages)
        
        raise NotImplementedError("Please implement OpenAI handler from your separate project")


class VendorFactory:
    """Factory for creating vendor handlers."""
    
    _handlers = {
        'gemini': GeminiHandler,
        'openai': OpenAIHandler,
    }
    
    @classmethod
    def get_vendor_handler(cls, vendor: str) -> VendorHandler:
        """
        Get vendor handler for the specified vendor.
        
        To integrate your existing handlers:
        1. Import your handler classes
        2. Replace the placeholder classes in _handlers dict
        3. Or override this method to use your existing factory
        """
        handler_class = cls._handlers.get(vendor.lower())
        if not handler_class:
            raise ValueError(f"Unsupported vendor: {vendor}")
        return handler_class()
    
    @classmethod
    def register_handler(cls, vendor: str, handler_class):
        """Register a custom vendor handler."""
        cls._handlers[vendor.lower()] = handler_class


class BatchProcessor:
    """Handles batching of content before vendor API calls."""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.vendor_handler = VendorFactory.get_vendor_handler(model_config.vendor)
    
    def process_batches(self, content_blocks: List[str], system_prompt: str = "") -> str:
        """
        Process content blocks in batches to optimize API calls.
        
        Args:
            content_blocks: List of content blocks to process
            system_prompt: System prompt for the LLM
            
        Returns:
            Combined result from all batches
        """
        if not content_blocks:
            return ""
        
        # Calculate system overhead from actual system prompt length
        # This uses the exact length of the system prompt instead of hardcoded values
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            system_overhead = len(encoding.encode(system_prompt)) if system_prompt else 0
        except:
            # Fallback to character-based estimation if tiktoken fails
            system_overhead = len(system_prompt) // 4 if system_prompt else 0
        
        # Add some buffer for message formatting overhead (JSON structure, etc.)
        formatting_overhead = 100  # Approximate tokens for message structure
        total_system_overhead = system_overhead + formatting_overhead
        
        # Calculate optimal batch size based on model capabilities
        max_input_tokens = self.model_config.context_window - self.model_config.output_window - total_system_overhead
        batch_size = max_input_tokens // (self.model_config.output_window * 1.5)
        
        instruction = "Summarize the following text while preserving all technical details, data, and semantic meaning. Maintain code blocks, JSON structures, and tables exactly as they appear:\n\n"
        
        summarized_results = []
        current_batch = []
        current_batch_tokens = 0
        
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            instruction_tokens = len(encoding.encode(instruction))
            
            for block in content_blocks:
                block_tokens = len(encoding.encode(block))
                
                # Check if adding this block would exceed context window
                # Include system prompt tokens in the calculation
                total_tokens_needed = current_batch_tokens + block_tokens + instruction_tokens + system_overhead
                if total_tokens_needed > max_input_tokens and current_batch:
                    # Process current batch
                    batch_result = self._process_batch(current_batch, instruction, system_prompt)
                    summarized_results.append(batch_result)
                    
                    # Start new batch
                    current_batch = [block]
                    current_batch_tokens = block_tokens
                else:
                    # Add to current batch
                    current_batch.append(block)
                    current_batch_tokens += block_tokens
            
            # Process final batch if exists
            if current_batch:
                batch_result = self._process_batch(current_batch, instruction, system_prompt)
                summarized_results.append(batch_result)
            
        except Exception:
            # Fallback to character-based estimation if tiktoken fails
            max_chars = max_input_tokens * 4
            current_batch_chars = 0
            
            for block in content_blocks:
                block_chars = len(block)
                
                # Include system prompt in character calculation
                total_chars_needed = current_batch_chars + block_chars + len(system_prompt) + len(instruction)
                if total_chars_needed > max_chars and current_batch:
                    # Process current batch
                    batch_result = self._process_batch(current_batch, instruction, system_prompt)
                    summarized_results.append(batch_result)
                    
                    # Start new batch
                    current_batch = [block]
                    current_batch_chars = block_chars
                else:
                    current_batch.append(block)
                    current_batch_chars += block_chars
            
            # Process final batch if exists
            if current_batch:
                batch_result = self._process_batch(current_batch, instruction, system_prompt)
                summarized_results.append(batch_result)
        
        # Combine all summarized results
        return "\n\n".join(summarized_results)
    
    def _process_batch(self, batch: List[str], instruction: str, system_prompt: str) -> str:
        """Process a single batch of content."""
        batch_content = "\n\n---\n\n".join(batch)
        
        messages = [{
            "role": "user",
            "content": instruction + batch_content
        }]
        
        # Add system prompt if provided
        if system_prompt:
            messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
        
        success, result, error_msg = self.vendor_handler.process_request(
            self.model_config,
            messages
        )
        
        if success:
            return result.content
        else:
            return f"[Batch processing failed: {error_msg}]\n{batch_content}"


class TokenCounter:
    """Handles token counting for different models."""
    
    @staticmethod
    def count_tokens(messages: List[Dict[str, str]], model_name: str) -> int:
        """
        Count tokens in a list of chat messages using tiktoken.
        
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
                "text-davinci-003": "p50k_base",
                "gemini-2.0-flash": "cl100k_base"
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


class ContentProcessor:
    """Handles content processing including protected sections and chunking."""
    
    @staticmethod
    def extract_protected_sections(text: str) -> List[Dict[str, Any]]:
        """Extract protected sections (code blocks, JSON, tables) from text."""
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
    
    @staticmethod
    def restore_protected_sections(text: str, protected_sections: List[Dict[str, Any]]) -> str:
        """Restore protected sections in summarized text."""
        result = text
        for section in protected_sections:
            placeholder = section['placeholder']
            if placeholder in result:
                result = result.replace(placeholder, section['content'])
        return result
    
    @staticmethod
    def chunk_text_optimized(text: str, max_tokens: int, output_tokens: int, system_prompt: str = "") -> List[str]:
        """Split text into optimized chunks based on model's context window."""
        if not text:
            return []
        
        # Calculate system overhead from actual system prompt length
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            system_overhead = len(encoding.encode(system_prompt)) if system_prompt else 0
        except:
            # Fallback to character-based estimation if tiktoken fails
            system_overhead = len(system_prompt) // 4 if system_prompt else 0
        
        # Add buffer for message formatting overhead
        formatting_overhead = 100
        total_system_overhead = system_overhead + formatting_overhead
        
        # Calculate optimal chunk size
        max_tokens_per_chunk = min(
            max_tokens - output_tokens - total_system_overhead,
            output_tokens * 4
        )
        
        # If text is small enough for single API call, return as-is
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            total_tokens = len(encoding.encode(text))
            if total_tokens <= max_tokens_per_chunk:
                return [text]
        except:
            if len(text) <= max_tokens_per_chunk * 4:
                return [text]
        
        # Use character-based chunking
        chars_per_chunk = max_tokens_per_chunk * 4
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chars_per_chunk
            
            # Try to find a natural break point
            if end < len(text):
                search_start = max(start, end - 500)
                sentence_break = text.rfind('.', search_start, end)
                if sentence_break > start:
                    end = sentence_break + 1
                else:
                    para_break = text.rfind('\n\n', search_start, end)
                    if para_break > start:
                        end = para_break + 2
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks


class ChatShrinker:
    """Main class for shrinking chat conversations."""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.batch_processor = BatchProcessor(model_config)
        self.token_counter = TokenCounter()
        self.content_processor = ContentProcessor()
    
    def shrink_chat(self, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Return an optimized chat list ≤ max_tokens while respecting all constraints.
        
        Args:
            chat_history: List of chat messages with 'role' and 'content' keys
            
        Returns:
            Optimized chat history that fits within token limit
        """
        if not chat_history:
            raise ValueError("Chat history cannot be empty")
        
        # Step 1: Copy the first system message exactly
        system_messages = [msg for msg in chat_history if msg.get("role") == "system"]
        if not system_messages:
            raise ValueError("No system message found in chat history")
        
        system_message = system_messages[0]
        
        # Step 2: Keep only the last user message
        user_messages = [msg for msg in chat_history if msg.get("role") == "user"]
        last_user_message = user_messages[-1] if user_messages else None
        
        # Step 3: Append all assistant messages
        assistant_messages = [msg for msg in chat_history if msg.get("role") == "assistant"]
        
        # Determine conversation flow order
        last_message = chat_history[-1] if chat_history else None
        conversation_ends_with_user = last_message and last_message.get("role") == "user"
        
        # Build provisional list maintaining proper conversation flow
        provisional_list = [system_message]
        
        if conversation_ends_with_user:
            provisional_list.extend(assistant_messages)
            if last_user_message:
                provisional_list.append(last_user_message)
        else:
            if last_user_message:
                provisional_list.append(last_user_message)
            provisional_list.extend(assistant_messages)
        
        # Step 4: Compute token count
        try:
            token_count = self.token_counter.count_tokens(provisional_list, self.model_config.name)
        except Exception as e:
            raise ValueError(f"Failed to count tokens: {str(e)}")
        
        # Step 5: If over the model's context window, start summarizing
        if token_count <= self.model_config.context_window:
            return provisional_list
        
        # Step 5.1-5.2: Summarize assistant messages first
        if assistant_messages:
            try:
                summarized_assistant = self._summarize_assistant_messages(assistant_messages)
                
                # Rebuild provisional list maintaining conversation flow order
                provisional_list = [system_message]
                
                if conversation_ends_with_user:
                    provisional_list.append(summarized_assistant)
                    if last_user_message:
                        provisional_list.append(last_user_message)
                else:
                    if last_user_message:
                        provisional_list.append(last_user_message)
                    provisional_list.append(summarized_assistant)
                
                token_count = self.token_counter.count_tokens(provisional_list, self.model_config.name)
            except Exception as e:
                raise ValueError(f"Failed to summarize assistant messages: {str(e)}")
        
        # Step 5.3: If still over budget, summarize the user message
        if token_count > self.model_config.context_window and last_user_message:
            try:
                summarized_user = self._summarize_user_message(last_user_message)
                
                # Rebuild with summarized user message
                provisional_list = [system_message]
                
                if conversation_ends_with_user:
                    if assistant_messages:
                        provisional_list.append(summarized_assistant)
                    provisional_list.append(summarized_user)
                else:
                    provisional_list.append(summarized_user)
                    if assistant_messages:
                        provisional_list.append(summarized_assistant)
                
                token_count = self.token_counter.count_tokens(provisional_list, self.model_config.name)
            except Exception as e:
                raise ValueError(f"Failed to summarize user message: {str(e)}")
        
        return provisional_list
    
    def _summarize_assistant_messages(self, assistant_messages: List[Dict[str, str]]) -> Dict[str, str]:
        """Summarize all assistant messages into a single high-fidelity summary."""
        if not assistant_messages:
            return {"role": "assistant", "content": ""}
        
        # Combine all assistant content while preserving protected sections
        combined_content = ""
        for i, msg in enumerate(assistant_messages):
            content = msg.get("content", "")
            if i > 0:
                combined_content += "\n\n---\n\n"
            combined_content += content
        
        # Extract and preserve protected sections
        protected_sections = self.content_processor.extract_protected_sections(combined_content)
        
        # Use optimized chunking with system prompt
        chunks = self.content_processor.chunk_text_optimized(
            combined_content, 
            self.model_config.context_window, 
            self.model_config.output_window,
            "You are a professional high-fidelity condensing protocol agent."
        )
        
        # Summarize chunks with batching
        try:
            summarized_blocks = self.batch_processor.process_batches(
                chunks, 
                "You are a professional high-fidelity condensing protocol agent."
            )
            
            # Restore protected sections in summary
            final_summary = self.content_processor.restore_protected_sections(
                summarized_blocks, 
                protected_sections
            )
            
            return {"role": "assistant", "content": final_summary}
        except Exception as e:
            raise Exception(f"Assistant message summarization failed: {str(e)}")
    
    def _summarize_user_message(self, user_message: Dict[str, str]) -> Dict[str, str]:
        """Summarize a user message while preserving technical fidelity."""
        content = user_message.get("content", "")
        
        # If content is already short, return as-is
        try:
            token_count = self.token_counter.count_tokens([user_message], self.model_config.name)
            if token_count <= 500:
                return user_message
        except:
            pass
        
        # Use optimized chunking strategy with system prompt
        chunks = self.content_processor.chunk_text_optimized(
            content, 
            self.model_config.context_window, 
            self.model_config.output_window,
            "You are a professional high-fidelity condensing protocol agent."
        )
        
        try:
            summarized_content = self.batch_processor.process_batches(
                chunks, 
                "You are a professional high-fidelity condensing protocol agent."
            )
            return {"role": "user", "content": summarized_content}
        except Exception as e:
            raise Exception(f"User message summarization failed: {str(e)}")


class ChatSummarizerApp:
    """Main application class for the chat summarizer."""
    
    def __init__(self):
        self.model_configs = {
            'gemini-2.0-flash': ModelConfig(
                name='gemini-2.0-flash',
                vendor='gemini',
                context_window=1000000,
                output_window=8000
            ),
            'gpt-4o': ModelConfig(
                name='gpt-4o',
                vendor='openai',
                context_window=128000,
                output_window=4000
            ),
            'gpt-4': ModelConfig(
                name='gpt-4',
                vendor='openai',
                context_window=32000,
                output_window=4000
            )
        }
    
    def process_chat(self, chat_history: List[Dict[str, str]], model_name: str) -> Dict[str, Any]:
        """
        Process a chat history with the specified model.
        
        Args:
            chat_history: List of chat messages
            model_name: Name of the model to use
            
        Returns:
            Dictionary containing the shrunk chat and metadata
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model_config = self.model_configs[model_name]
        shrinker = ChatShrinker(model_config)
        
        start_time = time.time()
        shrunk_chat = shrinker.shrink_chat(chat_history)
        end_time = time.time()
        
        # Calculate metadata
        original_tokens = shrinker.token_counter.count_tokens(chat_history, model_name)
        shrunk_tokens = shrinker.token_counter.count_tokens(shrunk_chat, model_name)
        reduction_percent = ((original_tokens - shrunk_tokens) / original_tokens * 100) if original_tokens > 0 else 0
        
        return {
            'shrunk_chat': shrunk_chat,
            'metadata': {
                'original_messages': len(chat_history),
                'shrunk_messages': len(shrunk_chat),
                'model_name': model_name,
                'model_context_window': model_config.context_window,
                'model_output_window': model_config.output_window,
                'processing_time_seconds': end_time - start_time,
                'estimated_original_tokens': original_tokens,
                'estimated_shrunk_tokens': shrunk_tokens,
                'token_reduction_percentage': reduction_percent,
                'timestamp': time.time()
            }
        }
    
    def save_result(self, result: Dict[str, Any], filename: str):
        """Save the result to a JSON file."""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': result['metadata'],
                'data': result['shrunk_chat']
            }, f, indent=2, ensure_ascii=False)


# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_chat = [
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
    
    print("=== CHAT SUMMARIZER OOP DEMO ===")
    print("Using Object-Oriented Programming with vendor abstraction and batching")
    print()
    
    app = ChatSummarizerApp()
    
    # Test with different models
    for model_name in ['gemini-2.0-flash', 'gpt-4o', 'gpt-4']:
        print(f"--- Testing with {model_name} ---")
        
        try:
            result = app.process_chat(test_chat, model_name)
            
            print(f"Original: {result['metadata']['original_messages']} messages, ~{result['metadata']['estimated_original_tokens']:,} tokens")
            print(f"Shrunk: {result['metadata']['shrunk_messages']} messages, ~{result['metadata']['estimated_shrunk_tokens']:,} tokens")
            print(f"Reduction: {result['metadata']['token_reduction_percentage']:.1f}%")
            print(f"Processing time: {result['metadata']['processing_time_seconds']:.2f} seconds")
            
            # Save result
            filename = f'shrunk_chats/shrunk_chat_{model_name.replace("-", "_")}_{result["metadata"]["model_context_window"]}.json'
            app.save_result(result, filename)
            print(f"Saved to: {filename}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
        
        print()
    
    print("Demo completed!")
    print("\nKey OOP Features:")
    print("  - Vendor abstraction with VendorHandler interface")
    print("  - Batch processing before API calls")
    print("  - Modular design with separate concerns")
    print("  - Easy to extend with new vendors")
    print("  - Comprehensive error handling")
