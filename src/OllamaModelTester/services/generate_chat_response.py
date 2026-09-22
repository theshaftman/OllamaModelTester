from typing import Any, Dict, List


class GenerateChatResponse():    
    def generate_chat_response(
        self, 
        model_name: str = None, 
        messages: List[Dict[str, Any]] = None,
        tools: List[Any] = None,
        imported_modules: Dict[str, Any] = None
    ) -> Any:
        response = None
        try:
            if messages is None:
                messages = []
            if tools is None:
                tools = []
            chat = imported_modules['chat']
            response = chat(
                model=model_name,
                messages=messages,
                tools=tools
            )
            messages.append(response.message.model_dump(exclude_none=True))
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return response, messages
