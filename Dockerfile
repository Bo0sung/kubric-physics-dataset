FROM kubricdockerhub/kubruntu:latest

USER root
RUN /usr/bin/python3 -m pip install --no-cache-dir imageio imageio-ffmpeg
WORKDIR /workspace

