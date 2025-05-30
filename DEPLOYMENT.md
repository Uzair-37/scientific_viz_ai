# Deployment Guide

This guide provides instructions for deploying the Multi-Modal AI web application publicly.

## Deployment Options

There are several options for deploying your Streamlit application:

1. **Streamlit Cloud** (Easiest)
2. **Heroku**
3. **AWS Elastic Beanstalk**
4. **Google Cloud Run**
5. **Azure App Service**
6. **Docker + VPS** (Digital Ocean, Linode, etc.)

Below are instructions for each approach.

## 1. Streamlit Cloud (Recommended for simplicity)

[Streamlit Cloud](https://streamlit.io/cloud) is the easiest way to deploy Streamlit apps. It's free for public repositories.

### Steps:

1. Push your code to a GitHub repository:
   ```bash
   # Initialize git if not already done
   git init
   git add .
   git commit -m "Initial commit"
   
   # Link to GitHub repo
   git remote add origin https://github.com/yourusername/scientific_viz_ai.git
   git push -u origin main
   ```

2. Sign up at [share.streamlit.io](https://share.streamlit.io/) using your GitHub account
3. Click "New app"
4. Select your repository, branch, and the path to the app file (`web_app/app.py`)
5. Click "Deploy"

Your app will be available at `https://share.streamlit.io/yourusername/scientific_viz_ai/main/web_app/app.py`

## 2. Heroku Deployment

### Prerequisites:
- Heroku account
- Heroku CLI installed

### Steps:

1. Create a `Procfile` in the project root:

```
web: sh setup.sh && streamlit run web_app/app.py
```

2. Create `setup.sh` in the project root:

```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

3. Make sure your `requirements.txt` is up to date:

```bash
pip freeze > requirements.txt
```

4. Initialize Heroku app:

```bash
heroku login
heroku create scientific-viz-ai
git push heroku main
```

Your app will be available at `https://scientific-viz-ai.herokuapp.com`

## 3. AWS Elastic Beanstalk

### Prerequisites:
- AWS account
- AWS CLI and EB CLI installed

### Steps:

1. Install EB CLI:

```bash
pip install awsebcli
```

2. Create a `Procfile` in the project root:

```
web: streamlit run web_app/app.py --server.port=$PORT --server.address=0.0.0.0
```

3. Create an `.ebignore` file (similar to `.gitignore`) to exclude unnecessary files.

4. Initialize EB application:

```bash
eb init -p python-3.8 scientific-viz-ai
```

5. Create environment and deploy:

```bash
eb create scientific-viz-ai-env
```

6. Your app will be available at the provided EB URL.

## 4. Google Cloud Run

### Prerequisites:
- Google Cloud account
- Google Cloud SDK installed
- Docker installed

### Steps:

1. Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "web_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and push the Docker image:

```bash
# Set your Google Cloud project ID
PROJECT_ID=your-project-id

# Build the container image
gcloud builds submit --tag gcr.io/$PROJECT_ID/scientific-viz-ai

# Deploy to Cloud Run
gcloud run deploy scientific-viz-ai \
  --image gcr.io/$PROJECT_ID/scientific-viz-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

Your app will be available at the provided Cloud Run URL.

## 5. Azure App Service

### Prerequisites:
- Azure account
- Azure CLI installed

### Steps:

1. Create a `Dockerfile` as shown in the Google Cloud Run section.

2. Deploy to Azure App Service:

```bash
# Login to Azure
az login

# Create a resource group
az group create --name scientific-viz-ai --location eastus

# Create an App Service plan
az appservice plan create --name scientific-viz-ai-plan --resource-group scientific-viz-ai --sku B1 --is-linux

# Create a web app
az webapp create --resource-group scientific-viz-ai --plan scientific-viz-ai-plan --name scientific-viz-ai --deployment-container-image-name docker.io/username/scientific-viz-ai:latest

# Configure the web app
az webapp config appsettings set --resource-group scientific-viz-ai --name scientific-viz-ai --settings PORT=8501
```

Your app will be available at `https://scientific-viz-ai.azurewebsites.net`

## 6. Docker + VPS (Digital Ocean, Linode, etc.)

### Prerequisites:
- VPS (Virtual Private Server)
- SSH access to the server
- Docker installed on the server

### Steps:

1. Create a `Dockerfile` as shown in the Google Cloud Run section.

2. Build and push the Docker image:

```bash
# Build the image
docker build -t scientific-viz-ai .

# (Optional) Push to Docker Hub
docker tag scientific-viz-ai username/scientific-viz-ai
docker push username/scientific-viz-ai
```

3. SSH into your VPS and run:

```bash
# Pull the image (if using Docker Hub)
docker pull username/scientific-viz-ai

# Run the container
docker run -d -p 80:8501 username/scientific-viz-ai
```

4. Configure your domain to point to your VPS IP address.

Your app will be available at your domain or the VPS IP address.

## Production Considerations

For a production deployment, consider the following:

### 1. Environment Variables

Store sensitive information like API keys in environment variables:

```python
import os

api_key = os.environ.get("API_KEY", "default_key")
```

For local development, you can use a `.env` file and the python-dotenv package.

### 2. Performance Optimization

- Use caching for expensive computations:
  ```python
  @st.cache_data
  def load_data():
      # Expensive computation
      return data
  ```

- Consider using session state to store user data:
  ```python
  if "data" not in st.session_state:
      st.session_state.data = load_data()
  ```

### 3. Security

- Validate user inputs
- Use HTTPS
- Consider adding authentication if needed (Streamlit Community Cloud supports this)

### 4. Monitoring and Logging

- Set up logging:
  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  ```

- Consider services like Sentry for error tracking

### 5. Scalability

For high-traffic applications, consider:
- Separating the model serving from the web application
- Using a dedicated model serving solution like TorchServe or TensorFlow Serving
- Implementing a caching layer with Redis

## Resources

- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/deploy)
- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [AWS Elastic Beanstalk Python Guide](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-apps.html)
- [Google Cloud Run Guide](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service)
- [Azure Web App for Containers](https://docs.microsoft.com/en-us/azure/app-service/quickstart-custom-container)