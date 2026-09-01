import os
import subprocess
import time


class ExecuteCommand():

    @staticmethod
    def run_command(command: str = '', check=True, timeout: int = 120):
        var_subprocess = None
        try:
            var_subprocess = subprocess.run(
                command,
                check=check,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            time.sleep(1)
            print(f'Command succeded: {command}')
            if (var_subprocess.stdout):
                print(f'Output: {var_subprocess.stdout}')
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return var_subprocess

    @staticmethod
    def run_popen(command: str = '', text: bool = True):
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
            time.sleep(1)
        except Exception as e:
            print(f'Exception thrown: {str(e)}')
        finally:
            return var_subprocess