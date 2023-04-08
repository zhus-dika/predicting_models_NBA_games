FROM conda/miniconda3
WORKDIR app
COPY models/salary_model.onnx models/
COPY src/models/salary_model.py models/
COPY src/app.py .
COPY environment.yml .
RUN conda update conda
RUN conda env create -f environment.yml
EXPOSE 5000
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nba", "python3", "app.py"]
