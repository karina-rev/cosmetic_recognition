FROM ubuntu:18.04

ARG PACKAGE=intel-openvino-dev-ubuntu18-2020.4.287

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y --no-install-recommends \
  ca-certificates \
  gnupg \
  wget

RUN wget https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020 && \
  apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2020

RUN echo deb "https://apt.repos.intel.com/openvino/2020 all main" > /etc/apt/sources.list.d/intel-openvino-2020.list

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    $PACKAGE && \
  apt autoremove -y && \
  rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "source /opt/intel/openvino/bin/setupvars.sh"

RUN echo "source /opt/intel/openvino/bin/setupvars.sh" >> /root/.bashrc

CMD ["/bin/bash"]