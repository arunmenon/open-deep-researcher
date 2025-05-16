import re
import json
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union, Type
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class ResponseParser:
    """Utility class for parsing and extracting structured data from LLM responses"""
    
    @staticmethod
    def parse_json_response(response_text: str, model_class: Type[T]) -> T:
        """Extract and parse JSON from a model response text
        
        Args:
            response_text: The raw text response from the model
            model_class: The Pydantic model class to parse into
            
        Returns:
            Instance of the specified model class
            
        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        try:
            # First try to find JSON pattern in the text
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON block found, try to parse the whole text
                json_str = response_text
            
            # Parse JSON into a dictionary
            json_data = json.loads(json_str)
            
            # Validate against the model schema
            return model_class(**json_data)
            
        except (json.JSONDecodeError, Exception) as e:
            # Try alternative extraction methods
            try:
                return ResponseParser._extract_structured_data(response_text, model_class)
            except Exception as inner_e:
                raise ValueError(f"Failed to parse response: {str(e)}. Inner error: {str(inner_e)}")
    
    @staticmethod
    def _extract_structured_data(text: str, model_class: Type[T]) -> T:
        """Extract structured data from text using field names from the model
        
        This is a fallback method when JSON parsing fails.
        It tries to extract data based on model field names and patterns.
        
        Args:
            text: The text to extract from
            model_class: The Pydantic model class
            
        Returns:
            Instance of the specified model class
        """
        # Get field names from the model
        field_names = list(model_class.__annotations__.keys())
        
        # Initialize result dictionary
        result = {}
        
        # For list fields
        for field_name in field_names:
            if field_name.endswith('s') and "List" in str(model_class.__annotations__[field_name]):
                # Try to find a section with this field name
                pattern = rf'{field_name}:?\s*(?:\n|$)(.*?)(?:(?:\n\s*\n)|$)'
                section_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                
                if section_match:
                    section_text = section_match.group(1)
                    # Extract list items (numbered or bulleted)
                    items = re.findall(r'(?:\d+\.|-|\*)\s*(.*?)(?=(?:\d+\.|-|\*)|$)', section_text, re.DOTALL)
                    items = [item.strip() for item in items if item.strip()]
                    if items:
                        result[field_name] = items
                
                # If no section found or no items extracted, check for inline lists
                if field_name not in result or not result[field_name]:
                    # Try to find "field_name: [item1, item2]" or "field_name: item1, item2"
                    inline_pattern = rf'{field_name}:?\s*\[?([^\[\]\n]+)\]?'
                    inline_match = re.search(inline_pattern, text, re.IGNORECASE)
                    if inline_match:
                        items_text = inline_match.group(1)
                        items = [item.strip() for item in items_text.split(',') if item.strip()]
                        if items:
                            result[field_name] = items
            
            # For boolean fields
            elif "bool" in str(model_class.__annotations__[field_name]):
                pattern = rf'{field_name}:?\s*(true|false|yes|no)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).lower()
                    result[field_name] = value in ['true', 'yes']
            
            # For numeric fields
            elif any(numeric_type in str(model_class.__annotations__[field_name]) for numeric_type in ["int", "float", "number"]):
                pattern = rf'{field_name}:?\s*(\d+(?:\.\d+)?\)?)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if "int" in str(model_class.__annotations__[field_name]):
                        result[field_name] = int(value)
                    else:
                        result[field_name] = float(value)
            
            # For string fields
            elif "str" in str(model_class.__annotations__[field_name]):
                pattern = rf'{field_name}:?\s*"?([^"\n]+)"?'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result[field_name] = match.group(1).strip()
        
        # Apply default values for missing required fields
        for field_name in field_names:
            if field_name not in result:
                # Set default values based on field type
                if "List" in str(model_class.__annotations__[field_name]):
                    result[field_name] = []
                elif "bool" in str(model_class.__annotations__[field_name]):
                    # Default boolean field to False
                    result[field_name] = False
                elif any(numeric_type in str(model_class.__annotations__[field_name]) for numeric_type in ["int", "float"]):
                    # Default numeric field to 0
                    if "int" in str(model_class.__annotations__[field_name]):
                        result[field_name] = 0
                    else:
                        result[field_name] = 0.0
                elif "str" in str(model_class.__annotations__[field_name]):
                    # Default string field to empty string
                    result[field_name] = ""
        
        # Create model instance
        return model_class(**result)