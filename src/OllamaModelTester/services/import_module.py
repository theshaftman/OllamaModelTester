
import importlib
from typing import Dict, Any

from .execute_cmd import ExecuteCommand as ec


class ImportModule():

    REQUIRED_VERSIONS = {
        'numpy': { 'import': 'numpy', 'version': '2.1.3', 'light_version': None },
        'torch': { 'import': 'torch', 'version': '2.11.0', 'light_version': '--index-url https://download.pytorch.org/whl/cpu' },
        'transformers': { 'import': 'transformers', 'version': '5.15.0', 'light_version': None },
        'ollama': { 'import': 'ollama', 'version': '0.6.2', 'light_version': None },
        'nltk': { 'import': 'nltk', 'version': '3.9.1', 'light_version': None },
        'scikit-learn': { 'import': 'sklearn', 'version': '1.6.1', 'light_version': None },
        'pandas': { 'import': 'pandas', 'version': '2.2.3', 'light_version': None },
        'matplotlib': { 'import': 'matplotlib', 'version': '3.10.0', 'light_version': None },
        'google-auth': { 'import': 'google', 'version': '2.49.0', 'light_version': None },
        'google-cloud-bigquery': { 'import': 'google.cloud', 'version': '3.43.0', 'light_version': None },
        'pandas-gbq': { 'import': 'pandas_gbq', 'version': '0.30.0', 'light_version': None },
        'google-api-core': { 'import': 'google.api_core', 'version': '2.30.3', 'light_version': None },
        'requests': { 'import': 'requests', 'version': '2.32.4', 'light_version': None },
        'nest-asyncio': { 'import': 'nest_asyncio', 'version': '1.6.0', 'light_version': None },
        'typing': { 'import': 'typing', 'version': '3.7.4.3', 'light_version': None },
        'pyarrow': { 'import': 'pyarrow', 'version': '25.0.1', 'light_version': None }
    }

    @staticmethod
    def import_all(is_libraries_exec_requested: bool = True, timeout: int = 120) -> Dict[str, Any]:
        loaded_modules = {}
        try:
            for package_name, package_data in ImportModule.REQUIRED_VERSIONS.items():
                required_import = package_data['import']
                required_version = package_data['version']
                required_light_version = package_data['light_version']

                if (is_libraries_exec_requested):
                    timeout = 600 if package_name == 'torch' and timeout < 600 else timeout
                    var_command = f'pip install {package_name}=={required_version} {required_light_version or ''}'
                    try:
                        ec.run_command(command=var_command, check=True, timeout=timeout)
                    except Exception as e:
                        print(f'Execption thrown: {str(e)}')

                module = importlib.import_module(required_import)
                loaded_modules[required_import] = module
        except Exception as e:
            print(f'Execption thrown: {str(e)}')
        finally:
            return loaded_modules
