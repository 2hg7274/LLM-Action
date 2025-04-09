import os
import psutil
import json

# 1. 시스템 프로세스 정보를 반환하는 함수
def get_top_processes_by_memory():
    """
    This tool allows real-time retrieval of the top 10 running processes sorted in descending order by memory usage percentage.
    It uses psutil to collect process information and returns the results in JSON format.
    """


    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    processes = sorted(processes, key=lambda proc: proc.get('memory_percent', 0), reverse=True)
    top_processes = processes[:10]
    return json.dumps(top_processes, indent=2)


# 2. 프로세스를 종료하는 함수를 구현합니다.
def kill_process(pid: int) -> str:
    """
    This tool terminates a Linux program given its process ID (PID).
    It uses psutil to locate the process and sends a termination signal.
    If the process does not exit gracefully, it forcefully kills the process.
    [Note]
    - It will never terminate the currently running process.
    - This tool is intended to terminate Python processes only.
    """


    current_pid = os.getpid()
    if pid == current_pid:
        return f"Cannot terminate the currently running process (PID {current_pid})."
    
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return f"No process found with PID {pid}."
    
    try:
        # Attempt to terminate gracefully
        process.terminate()
        process.wait(timeout=3)
    except psutil.TimeoutExpired:
        # Force kill if graceful termination fails
        process.kill()
        process.wait(timeout=3)
    except Exception as e:
        return f"An error occurred while terminating the process: {str(e)}"
    
    return f"Process with PID {pid} has been terminated successfully."