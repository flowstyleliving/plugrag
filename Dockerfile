FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY plugrag ./plugrag
COPY lens_rag ./lens_rag
# Use CPU wheels so a portable image does not pull CUDA libraries.
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir '.[server,llm]'
COPY examples/documents.json /examples/documents.json
ENV RAG_DATA_DIR=/data HF_HOME=/models
RUN useradd --create-home --uid 10001 rag && mkdir /data /models && chown rag:rag /data /models
USER rag
EXPOSE 8000
ENTRYPOINT ["plugrag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
