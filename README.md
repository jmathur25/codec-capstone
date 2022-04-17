# codec-capstone

## RLVC
We need to use a Docker image to reproduce the RLVC baseline. See `docker/rlvc.Dockerfile` for more explanation. If setting up for the first time:
```
cd docker
# builds Docker image to run RLVC
bash run.sh
sudo docker exec -it rlvc bash
# add your public key to ~/.ssh/authorized_keys
```

Normally, the RLVC Docker container should already be setup. You may need to add your key to `~/.ssh/authorized_key` in the container. Then, connect to the container:
```
ssh -p 8000 root@<SERVER IP>
```

