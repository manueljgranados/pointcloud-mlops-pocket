FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs

RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "pointcloud_mlops.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
