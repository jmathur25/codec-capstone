
set -eEou pipefail

# build the latest container image
sudo docker build . -f rlvc.Dockerfile -t rlvc

# run the container
# -d makes it run in the background
# --name gives the container the name rlvc. You can check on it by doing `docker ps -a`.
# --gpus gives it GPU access
# -v is a volume mount that maps files on the host to files in the container
# the volume mounts here give it access to the project and to binaries in /user/local/bin
# -p gives it an exposed port. You can then SSH into the Docker container! `ssh -p 8000 root@<host ip>`.
# We can now develop in the container remotely.
workdir=$(realpath ../..)
sudo docker run -d --name rlvc --gpus all -v $workdir:$workdir -p 8000:22 rlvc

echo "==========================================="
echo "You can access the container with:"
echo "sudo docker exec -it rlvc bash"
echo "Also, add your public key to ~/.ssh/authorized_keys to SSH into the container."
echo "==========================================="

