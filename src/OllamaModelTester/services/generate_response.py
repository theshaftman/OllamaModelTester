from typing import Any, Dict


class GenerateResponse():

    @staticmethod
    def generate(
        model_name: str = None,
        prompt_text: str = None,
        imported_modules: Dict[str, Any] = None,
        **options
    ) -> Any:
        response_generator = None
        var_options = {
            'seed': options.get('seed', None),
            'temperature': options.get('temperature', 0.1),
            'top_k': options.get('top_k', 40),
            'top_p': options.get('top_p', 0.9),
            'num_ctx': options.get('num_ctx', 512),
            'num_predict': options.get('num_predict', 128)
        }

        fingerprint = {
            'model': model_name,
            'prompt': prompt_text,
            'parameters': var_options,
            'response': {
                'text': None,
                'length': 0,
                'done': False,
                'finish_reason': None,
            },
            'performance': {
                'eval_count': None,
                'prompt_eval_count': None,
                'eval_duration': None,
                'prompt_eval_duration': None,
            }
        }
        try:
            ollama = imported_modules['ollama']
            response_generator = ollama.generate(
                model=model_name,
                prompt=prompt_text,
                options=var_options,
                stream=True
            )
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return response_generator, var_options, fingerprint
