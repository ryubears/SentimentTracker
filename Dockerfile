# Container image rather than a zip + layers: pandas, numpy and scipy together
# blow past the 250 MB unzipped limit for zip-packaged Lambdas.
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    && pip install --no-cache-dir boto3

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY run_period.py backfill.py lambda_handler.py config.yaml ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]
