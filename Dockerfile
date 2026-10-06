FROM kubricdockerhub/kubruntu:latest

USER root
RUN /usr/bin/python3 -m pip install --no-cache-dir imageio-ffmpeg==0.4.9
WORKDIR /workspace
