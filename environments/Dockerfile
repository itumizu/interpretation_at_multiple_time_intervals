FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG USER_NAME
ARG GROUP_ID
ARG GROUP_NAME
ARG PASSWORD
ARG CURRENT_DIR_PATH

ENV LANG C.UTF-8
ENV TZ Asia/Tokyo
ENV USER_ID $USER_ID
ENV USER_NAME $USER_NAME
ENV GROUP_ID $GROUP_ID
ENV GROUP_NAME $GROUP_NAME
ENV PASSWORD $PASSWORD
ENV CURRENT_DIR_PATH $CURRENT_DIR_PATH
ENV PATH /opt/conda/bin:$PATH

ENV NOTVISIBLE "in users profile"

RUN apt-get update --fix-missing && \
    apt-get -y upgrade && \
    apt-get install -y apt-utils && \
    apt-get install -y software-properties-common git emacs vim curl unzip htop openssh-server wget procps sudo nodejs

COPY scripts/entrypoint.sh /var/tmp
RUN chmod a+x /var/tmp/entrypoint.sh
RUN /var/tmp/entrypoint.sh

RUN usermod -G sudo ${USER_NAME}
RUN /usr/bin/ssh-keygen -A

RUN mkdir -p $CURRENT_DIR_PATH
WORKDIR $CURRENT_DIR_PATH

COPY ./etc/home/.ssh/ /home/$USER_NAME/.ssh
COPY python/requirements.base.txt $CURRENT_DIR_PATH

RUN pip install --upgrade pip
RUN pip install -r requirements.base.txt

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]