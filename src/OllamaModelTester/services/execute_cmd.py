import os
import subprocess
import time
from typing import Any


class ExecuteCommand():

    @staticmethod
    def run_command(command: str = '', check=True, timeout: int = 120) -> Any:
        var_subprocess = None
        try:
            var_subprocess = subprocess.run(
                command,
                check=check,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout
            )
            print(f'Command succeded: {command}')
            if (var_subprocess.stdout and check):
                print(f'Output: {var_subprocess.stdout}')
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return var_subprocess

    @staticmethod
    def run_popen(command: str = '', text: bool = True) -> Any:
        var_subprocess = None
        try:
            command = command.split()
            var_subprocess = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ,
                shell=False,
                text=text,
                encoding='utf-8',
                errors='replace'
            )
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return var_subprocess