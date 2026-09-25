FROM mathmodel-ai-solver:phase4

# Autopilot execution image: adds spreadsheet output and figure support on
# top of the existing networkless, non-root solver sandbox image.
USER root
RUN pip install --no-cache-dir openpyxl matplotlib
