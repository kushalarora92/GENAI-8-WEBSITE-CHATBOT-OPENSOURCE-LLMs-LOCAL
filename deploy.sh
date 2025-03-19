#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

IMAGE_NAME=$1
IMAGE_TAG=$2
APP_PORT=$3

echo "Deploying ${IMAGE_NAME}:${IMAGE_TAG} on port ${APP_PORT}"

# Stop and remove the container if it exists
echo "Stopping and removing existing container with name ${IMAGE_NAME}..."
docker stop ${IMAGE_NAME} || true
docker rm ${IMAGE_NAME} || true

# Load the Docker image from the tar file
echo "Loading new image..."
docker load < image.tar

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    exit 1
fi

# Ensure static directory exists with proper permissions
echo "Setting up static directory..."
mkdir -p ~/static/docs
sudo chown -R 1000:1000 ~/static  # Match container's user ID
chmod -R 755 ~/static  # More secure permissions

# Run the new container
echo "Starting new container..."
docker run -d \
  --name ${IMAGE_NAME} \
  --restart unless-stopped \
  --env-file ./.env \
  --cpus="0.5" \
  --memory="500m" \
  --memory-swap="500m" \
  -p ${APP_PORT}:${APP_PORT} \
  -v ~/static:/app/static \
  ${IMAGE_NAME}:${IMAGE_TAG}

docker stop ai7-custom-website-chatbot

# docker start ai7-custom-website-chatbot
# docker update --cpus="0.5" --memory="500m" --memory-swap="500m" ai7-custom-website-chatbot
# docker update --cpus="0.2" --memory="200m" --memory-swap="200m" ai6-interview-questions

# Verify the container is running
echo "Verifying container status..."
if ! docker ps | grep -q ${IMAGE_NAME}; then
    echo "Container failed to start!"
    docker logs ${IMAGE_NAME}
    exit 1
fi

# Wait a bit and check if container is still running (in case it exits quickly)
sleep 5
if ! docker ps | grep -q ${IMAGE_NAME}; then
    echo "Container stopped shortly after starting!"
    docker logs ${IMAGE_NAME}
    exit 1
fi

# Clean up
rm image.tar

echo "Deployment complete!" 