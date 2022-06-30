#!/bin/bash

USER_NAME=$(id -un)
USER_ID=$(id -u)
GROUP_NAME=$(id -gn)
GROUP_ID=$(id -g)
CURRENT_DIR_PATH=$(pwd)

echo `pwd`
source .env

if [ ! -e ./environments/etc/ssh/ssh_host_rsa_key ]; then
  ssh-keygen -t rsa -N '' -f ./environments/etc/ssh/ssh_host_rsa_key
fi

if [ ! -e ./environments/etc/ssh/ssh_host_dsa_key ]; then
  ssh-keygen -t dsa -N '' -f ./environments/etc/ssh/ssh_host_dsa_key
fi

if [ ! -e ./environments/etc/ssh/ssh_host_ed25519_key ]; then
  ssh-keygen -t ed25519 -N '' -f ./environments/etc/ssh/ssh_host_ed25519_key
fi

echo ""
echo "------------------------------------------------------"
echo "1. ***** Input user password *****"

SALT=`python3 -c "import crypt; print(crypt.mksalt())"`

read -sp "Enter password for the user in the container.: " PASSWORD
tty -s && echo

hash_password=`python3 -c "import crypt, getpass, pwd, sys; print(crypt.crypt(sys.argv[1], sys.argv[2]))" $PASSWORD $SALT` 

read -sp "Enter same password again for the user in the container.: " PASSWORD_REPEAT
hash_password_repeat=`python3 -c "import crypt, getpass, pwd, sys; print(crypt.crypt(sys.argv[1], sys.argv[2]))" $PASSWORD_REPEAT $SALT` 

is_same_hash=`python3 -c "import sys; from hmac import compare_digest; print(compare_digest(sys.argv[1], sys.argv[2]))" $hash_password $hash_password_repeat`

echo ""
echo "------------------------------------------------------"
echo "2. ***** Build Docker image *****"
echo ""

if [ $is_same_hash ]
then
    docker image build \
      ./environments/ \
      --tag ${IMAGE_TAG} \
      --build-arg USER_NAME=${USER_NAME} \
      --build-arg USER_ID=${USER_ID} \
      --build-arg GROUP_NAME=${GROUP_NAME} \
      --build-arg GROUP_ID=${GROUP_ID} \
      --build-arg CURRENT_DIR_PATH=${CURRENT_DIR_PATH} \
      --build-arg PASSWORD=${hash_password}
    
    echo " ****** Build completed ******"

else
  echo "Passwords entered did not match. Try again."
  exit 1
fi