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


# 3. CPU 사용량 조회 도구
def get_cpu_usage() -> str:
    """
    Returns the current system CPU usage as a percentage.
    Uses psutil.cpu_percent(interval=1) to measure CPU usage over a 1-second interval.
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    return f"The current CPU usage is {cpu_percent}%."


# 4. 디스크 사용량 조회 도구
def get_disk_usage() -> str:
    """
    Retrieves disk usage statistics for the root filesystem.
    Uses psutil.disk_usage('/') to obtain total, used, free space, and usage percentage.
    Returns the statistics in JSON format.
    """
    disk = psutil.disk_usage('/')
    stats = {
        "total": disk.total,
        "used": disk.used,
        "free": disk.free,
        "percent": disk.percent
    }
    return json.dumps(stats, indent=2)



# 5. 메모리 정보 조회 도구
def get_memory_info() -> str:
    """
    Retrieves the current virtual memory statistics.
    Uses psutil.virtual_memory() to get total, available, used memory and the usage percentage.
    Returns the information in JSON format.
    """
    mem = psutil.virtual_memory()
    stats = {
        "total": mem.total,
        "available": mem.available,
        "used": mem.used,
        "percent": mem.percent
    }
    return json.dumps(stats, indent=2)



# 6. 시스템 에러 로르 조회 도구
def get_system_error_logs(log_file: str = "/var/log/syslog", num_lines: int = 10) -> str:
    """
    Retrieves the last few error log lines from a specified system log file.
    Filters lines that contain the word 'error' (case insensitive) and returns the last num_lines lines.
    If an error occurs during file reading, an error message is returned.
    """
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            error_lines = [line for line in lines if "error" in line.lower()]
            return "\n".join(error_lines[-num_lines:])
    except Exception as e:
        return f"An error occurred while reading the log file: {str(e)}"



# 7. 서비스 재시작 도구
def restart_service(service_name: str) -> str:
    """
    Restarts a specified Linux service.
    Uses the systemctl command to restart the service and returns the command output.
    Note: This function requires sudo permissions.
    """
    try:
        output = os.popen(f"sudo systemctl restart {service_name}").read()
        return f"Service '{service_name}' has been restarted. Output: {output}"
    except Exception as e:
        return f"An error occurred while restarting the service '{service_name}': {str(e)}"