# RAG Food & Wine Pairing App

This project leverages Retrieval-Augmented Generation (RAG) techniques and a set of NLP models to help answer food and wine-related questions. It uses AWS Bedrock-compatible models, Docker for deployment, and supports interactive exploration through Jupyter notebooks.

---

## 📁 Project Structure

```

image/
│
├── src/
│   ├── chroma\_db/                     # Chroma vectorstore database
│   ├── data/resources/
│   │   ├── food\_data.csv              # CSV of food-related data
│   │   ├── wine\_data.csv              # CSV of wine-related data
│   │   └── wine\_food\_pairing\_knowledge.pdf  # Domain knowledge PDF
│   ├── rag\_setup/
│   │   ├── get\_chroma\_db.py           # Load Chroma DB from vectorstore
│   │   ├── get\_embedding.py           # Load embedding model
│   │   └── get\_rag\_setup.py           # Configure paths and models
│   ├── spacy\_models/
│   │   ├── food\_intent\_model/         # Classifies if query is about food pairing or descriptions
│   │   ├── ner\_food\_model/            # Extracts food names and prices
│   │   ├── ner\_wine\_model/            # Extracts wine names, types, and prices
│   │   ├── textcat\_model/             # Classifies whether query is about food or wine
│   │   └── wine\_intent\_model/         # Detects wine pairing/description queries
│   ├── main.py                        # Main app script
│   └── query\_model.py                 # RAG-based query handler
│
├── .env                               # Sample AWS credentials file
├── Dockerfile                         # Docker build file for deployment
├── api\_app\_handler.py                 # API endpoints for Docker/AWS Lambda/EC2
├── requirements.txt                   # Python dependencies
├── vectorestore\_setup.py              # Creates vectorstore from Google Docs and PDFs
│                                      # (requires Google Auth credentials)
│
└── rag-env.ipynb                      # Jupyter Notebook: Interactive RAG pipeline walkthrough

````

---

## Assumptions

- You have an AWS account and **Bedrock access** configured.
- You have **AWS CLI installed and configured** via:
  ```bash
  aws configure
  ```

* AWS bootstrap and key pair generation are already done.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-user/your-repo.git
cd your-repo/image
```

---

### 2. Build and Test Docker Image Locally

```bash
docker build -t <your-image-name> .
docker run -d -p 8000:8000 <your-image-name>
docker ps  # Confirm the container is running
```

> Open your browser to `http://localhost:8000` to verify.

---

### 3. Push Docker Image to Docker Hub

1. Log in to Docker Hub:

   ```bash
   docker login
   ```
2. Tag and push your image:

   ```bash
   docker tag <your-image-name> your-dockerhub-user/<your-image-name>
   docker push your-dockerhub-user/<your-image-name>
   ```

---

### 4. Configure AWS EC2 for Deployment

1. Log in to your AWS Console.
2. Launch an **EC2 instance** and generate a **key pair**.
3. Ensure your **VPC is publicly accessible**:

   * Create subnets
   * Modify route tables:

     * Allow inbound rules:

       * SSH (TCP 22)
       * HTTP (TCP 80, 8000)
       * HTTPS (TCP 443)
   * Create and attach an **Internet Gateway**
   * Allocate and attach an **Elastic IP** for a static public IP

---

### 5. Connect to EC2 Instance via SSH

```bash
cd /path/to/keypair
chmod 400 your-key.pem
ssh -i "your-key.pem" ec2-user@<your-ec2-public-ip>
```

---

### 6. Set Up Docker on EC2

```bash
# Install Docker
sudo dnf install docker -y

# Enable Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Add user to Docker group
sudo usermod -aG docker ec2-user
exit  # Log out and log back in for group changes to take effect
```

---

### 7. Run Your App on EC2

```bash
# Pull your image
docker pull your-dockerhub-user/<your-image-name>

# Run container and map port 80 to 8000
docker run -d -p 80:8000 your-dockerhub-user/<your-image-name>
```

> Visit `http://<your-ec2-public-ip>` from your browser to access the API.

---

## 📓 Notebook Guide

Run `rag-env.ipynb` for a guided walkthrough of the entire RAG pipeline, including:

* Data ingestion
* Vectorstore creation
* Embedding setup
* Querying the pipeline with examples

---

## Notes

* For `vectorestore_setup.py`, you’ll need valid **Google API credentials** to access Google Docs.
* Make sure `.env` is properly set with your AWS access credentials for Bedrock model usage.
* Container is compatible with **AWS Lambda** if deployed with a handler from `api_app_handler.py`.

---

## Technologies Used

* **Spacy** for custom NLP models
* **LangChain + ChromaDB** for RAG
* **Docker** for containerization
* **AWS EC2 & Bedrock** for cloud deployment and model inference
* **FastAPI** for API integration

---

## Contact

For questions or collaboration, feel free to reach out at \[[kinnmrqz@gmail.com](mailto:kinnmrqz@gmail.com)].

---

