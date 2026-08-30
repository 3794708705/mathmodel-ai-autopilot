FROM python:3.12-slim
RUN pip install --no-cache-dir numpy scipy pandas matplotlib
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /workspace