FROM python:3.10.4-slim
RUN apt-get -y update && apt-get -y install git
RUN pip install --no-cache-dir poetry==1.5.1
WORKDIR nba
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install --no-root --no-directory --no-cache
COPY src/app/ src/app/
COPY src/salary/ src/salary/
RUN poetry check && poetry install --no-interaction --no-cache --compile --without dev
COPY models/ models/
COPY data/raw/advanced_plus_totals/ data/raw/advanced_plus_totals/
COPY dags/salary/ dags/salary/
RUN dvc init --no-scm
EXPOSE 5000
ENTRYPOINT ["python3", "-m", "src.app.app"]