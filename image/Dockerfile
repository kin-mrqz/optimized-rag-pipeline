FROM public.ecr.aws/lambda/python:3.13

# Install gcc, g++, and make to support builds
RUN dnf install -y gcc gcc-c++ make

# Upgrade pip to latest to access newest wheels
RUN python3 -m pip install --upgrade pip

RUN pip install spacy

# Copy requirements file into Lambda task root
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install pysqlite3-binary first if needed
RUN pip install pysqlite3-binary

# Install numpy and pandas first to ensure their wheels are used
RUN pip install --upgrade numpy pandas

# Then install remaining dependencies including spacy
RUN pip install --upgrade -r requirements.txt

# for local testing
EXPOSE 8000 

ENV IS_USING_IMAGE_RUNTIME=True

# Copy the source code into the container
# Ensure the directory structure matches the Lambda function's expectations
COPY "src/*" ${LAMBDA_TASK_ROOT}
COPY "src/chroma_db" ${LAMBDA_TASK_ROOT}/chroma_db
COPY "src/spacy_models" ${LAMBDA_TASK_ROOT}/spacy_models
COPY "src/data/resources/" ${LAMBDA_TASK_ROOT}/data/resources/
COPY "src/rag_setup" ${LAMBDA_TASK_ROOT}/rag_setup
COPY "src/rag_setup/get_chroma_db.py" ${LAMBDA_TASK_ROOT}/rag_setup/get_chroma_db.py
COPY "src/rag_setup/get_embedding.py" ${LAMBDA_TASK_ROOT}/rag_setup/get_embedding.py
COPY "src/rag_setup/get_rag_setup.py" ${LAMBDA_TASK_ROOT}/rag_setup/get_rag_setup.py
COPY "src/main.py" ${LAMBDA_TASK_ROOT}/main.py