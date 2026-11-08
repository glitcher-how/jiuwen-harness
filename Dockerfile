FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRAJ_PATH=/tmp/trajectories.jsonl \
    SUMMARY_PATH=/tmp/run_summary.json

WORKDIR /app
COPY requirements.lock .
RUN pip install -r requirements.lock

COPY run_agent.py ./
COPY harness ./harness

ENTRYPOINT ["python", "/app/run_agent.py"]
CMD ["--tasks-dir", "/data/tasks"]
