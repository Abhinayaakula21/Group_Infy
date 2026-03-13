import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT_OPTIONS = {
    "Milestone 1 (Hand Detection)": "milestone1.py",
    "Milestone 2 (Gesture + Distance)": "milestone2.py",
    "Milestone 3 (Volume + Graph)": "milestone3.py",
}

NOISY_LOG_SNIPPETS = (
    "Created TensorFlow Lite XNNPACK delegate for CPU.",
    "All log messages before absl::InitializeLog() is called are written to STDERR",
    "inference_feedback_manager.cc:114",
    "SymbolDatabase.GetPrototype() is deprecated.",
    "warnings.warn('SymbolDatabase.GetPrototype() is deprecated.",
)


def _init_state():
    st.session_state.setdefault("runner_process", None)
    st.session_state.setdefault("runner_queue", queue.Queue())
    st.session_state.setdefault("runner_logs", deque(maxlen=300))
    st.session_state.setdefault("runner_thread", None)


def _is_running() -> bool:
    proc = st.session_state.runner_process
    return proc is not None and proc.poll() is None


def _enqueue_process_output(proc: subprocess.Popen, out_queue: queue.Queue):
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        clean_line = line.rstrip()
        if any(snippet in clean_line for snippet in NOISY_LOG_SNIPPETS):
            continue
        out_queue.put(clean_line)
    proc.stdout.close()


def _drain_logs():
    while True:
        try:
            line = st.session_state.runner_queue.get_nowait()
        except queue.Empty:
            break
        st.session_state.runner_logs.append(line)


def _start_script(script_name: str):
    if _is_running():
        st.warning("A milestone script is already running. Stop it first.")
        return

    script_path = PROJECT_DIR / script_name
    if not script_path.exists():
        st.error(f"Script not found: {script_name}")
        return

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    t = threading.Thread(target=_enqueue_process_output, args=(proc, st.session_state.runner_queue), daemon=True)
    t.start()

    st.session_state.runner_process = proc
    st.session_state.runner_thread = t
    st.session_state.runner_logs.append(f"Started: {script_name} (PID {proc.pid})")


def _sync_process_state():
    proc = st.session_state.runner_process
    if proc is None:
        return

    code = proc.poll()
    if code is None:
        return

    st.session_state.runner_logs.append(f"Process exited with code: {code}")
    st.session_state.runner_process = None
    st.session_state.runner_thread = None


def _stop_script():
    proc = st.session_state.runner_process
    if proc is None:
        st.info("No running script to stop.")
        return

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
        st.session_state.runner_logs.append("Stopped running script.")
    else:
        st.session_state.runner_logs.append("Script already exited.")

    st.session_state.runner_process = None
    st.session_state.runner_thread = None


def main():
    st.set_page_config(page_title="Gesture Volume Controller", layout="wide")
    _init_state()
    _drain_logs()
    _sync_process_state()
    st.title("Gesture Volume Controller")
    st.write(
        "Use this page to run milestone scripts from one place. "
        "When a script starts, it will open its own OpenCV window."
    )

    with st.sidebar:
        st.header("Runner")
        selected_label = st.selectbox("Choose milestone", list(SCRIPT_OPTIONS.keys()))
        selected_script = SCRIPT_OPTIONS[selected_label]
        st.caption(f"File: {selected_script}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start", use_container_width=True):
                _start_script(selected_script)
        with col2:
            if st.button("Stop", use_container_width=True):
                _stop_script()

    status_col, help_col = st.columns([1, 2])
    with status_col:
        st.subheader("Status")
        if _is_running():
            proc = st.session_state.runner_process
            st.success(f"Running (PID {proc.pid})")
        else:
            st.info("Idle")

    with help_col:
        st.subheader("How To Use")
        st.markdown(
            "1. Choose a milestone in the sidebar.\n"
            "2. Click **Start**.\n"
            "3. Interact in the OpenCV window.\n"
            "4. Press `q` inside the OpenCV window to quit, or click **Stop** here."
        )

    st.subheader("Output Log")
    log_col1, log_col2 = st.columns(2)
    with log_col1:
        if st.button("Refresh Log", use_container_width=True):
            _drain_logs()
    with log_col2:
        if st.button("Clear Log", use_container_width=True):
            st.session_state.runner_logs.clear()
            st.session_state.runner_logs.append("Log cleared.")

    if st.button("Refresh Status"):
        _drain_logs()
        _sync_process_state()

    log_text = "\n".join(st.session_state.runner_logs) or "No output yet."
    st.text_area("Console", value=log_text, height=320)


if __name__ == "__main__":
    main()