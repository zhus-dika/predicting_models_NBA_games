FROM python:3.10.4-slim
RUN pip install --no-cache-dir poetry==1.5.1
WORKDIR nba
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install --no-root --no-directory --no-cache
COPY src/app/ src/app/
COPY src/salary/ src/salary/
COPY models/ models/
RUN poetry check && poetry install --no-interaction --no-cache --compile --without dev
EXPOSE 5000
ENTRYPOINT ["python3", "-m", "src.app.app"]