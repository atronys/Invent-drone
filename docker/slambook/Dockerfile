# Среда для упражнений slambook2 (Gao & Zhang, «Introduction to Visual SLAM»).
# Книга тестировалась на Ubuntu 18.04; на 24.04 код ломается на версиях Sophus/fmt/g2o/Pangolin,
# поэтому упражнения изолированы в контейнере, а основная система остаётся чистой.
#
# ЧЕРНОВИК — не проверялся сборкой (задача ENV-05). Если шаг падает — смотри issues репозитория
# slambook2; не трать на это больше одного вечера: упражнения можно делать и частично.
#
#   docker build -t slambook docker/slambook
#   xhost +local:docker
#   docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -v $PWD/learning:/work slambook
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates pkg-config \
      libeigen3-dev libopencv-dev libpcl-dev liboctomap-dev octovis \
      libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev libcxsparse3 \
      libglew-dev libgl1-mesa-dev libboost-all-dev libqglviewer-dev-qt5 \
    && rm -rf /var/lib/apt/lists/*

# 3rdparty в репозитории книги — подмодули с совместимыми версиями библиотек
RUN git clone --recursive --depth 1 https://github.com/gaoxiang12/slambook2.git /opt/slambook2
WORKDIR /opt/slambook2/3rdparty
RUN for d in Pangolin Sophus ceres-solver g2o DBow3; do \
      [ -d "$d" ] || continue; \
      cmake -S "$d" -B "$d/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
      cmake --build "$d/build" -j"$(nproc)" && cmake --install "$d/build" || exit 1; \
    done && ldconfig

WORKDIR /opt/slambook2
CMD ["bash"]
