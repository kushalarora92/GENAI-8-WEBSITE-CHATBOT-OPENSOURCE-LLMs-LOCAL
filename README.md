Retrieval Augmented Generation (RAG)

# Deployment
App is NOT deployed to EC2: We are using SSH Port Forwarding to local
```
ssh -i <pem_file> -R <local_port>:localhost:<app_port> <user>@<ec2-public-ip>
```
