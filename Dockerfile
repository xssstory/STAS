FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-10.0 \
    OPENMPI_VERSIONBASE=1.10
ENV OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.3
ENV OPENMPI_STRING=openmpi-${OPENMPI_VERSION} \
    OFED_VERSION=4.2-1.2.0.0

RUN mkdir -p $STAGE_DIR

RUN apt-get -y update && \
    apt-get -y install \
      build-essential \
      autotools-dev \
      rsync \
      curl \
      wget \
      jq \
      openssh-server \
      openssh-client \
    # No longer in 'minimal set of packages'
      sudo \
    # Needed by OpenMPI
      cmake \
      g++ \
      gcc \
    # ifconfig
      net-tools && \
    apt-get autoremove

WORKDIR $STAGE_DIR

# Install Mellanox OFED user-mode drivers and its prereqs
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    # For MLNX OFED
        dnsutils \
        pciutils \
        ethtool \
        lsof \
        python-libxml2 \
        quilt \
        libltdl-dev \
        dpatch \
        autotools-dev \
        graphviz \
        autoconf \
        chrpath \
        swig \
        automake \
        tk8.4 \
        tcl8.4 \
        libgfortran3 \
        tcl \
        libnl-3-200 \
        libnl-route-3-200 \
        libnl-route-3-dev \
        libnl-utils \
        gfortran \
        tk \
        bison \
        flex \
        libnuma1 \
        checkinstall && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    # libnl1 is not available in ubuntu16 so build from source
    wget -q -O - http://www.infradead.org/~tgr/libnl/files/libnl-1.1.4.tar.gz | tar xzf - && \
    cd libnl-1.1.4 && \
    ./configure && \
    make && \
    checkinstall -D --showinstall=no --install=yes -y -pkgname=libnl1 -A amd64 && \
    cd .. && \
    rm -rf libnl-1.1.4 && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu16.04-x86_64/DEBS && \
    for dep in libibverbs1 libibverbs-dev ibverbs-utils libmlx4-1 libmlx5-1 librdmacm1 librdmacm-dev libibumad libibumad-devel libibmad libibmad-devel; do \
        dpkg -i $dep\_*_amd64.deb; \
    done && \
    cd ../.. && \
    rm -rf MLNX_OFED_LINUX-*

##################### OPENMPI #####################

RUN wget -q -O - https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/${OPENMPI_STRING}.tar.gz | tar -xzf - && \
    cd ${OPENMPI_STRING} && \
    ./configure --prefix=/usr/local/${OPENMPI_STRING} && \
    make -j"$(nproc)" install && \
    rm -rf $STAGE_DIR/${OPENMPI_STRING} && \
    ln -s /usr/local/${OPENMPI_STRING} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++

# Update environment variables
ENV PATH=/usr/local/mpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends --allow-downgrades \
         git \
         ca-certificates locales \
         libnccl2=2.4.2-1+cuda10.0 \
         libjpeg-dev \
         libpng-dev &&\
         locale-gen en_US.UTF-8 && \
         rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN locale

ENV PYTHON_VERSION=3.6
ENV PATH /usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -L  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl tqdm&& \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION pytorch=1.2.0 torchvision -c pytorch && \
    /opt/conda/bin/conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace

RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION nltk && \
    /opt/conda/bin/conda clean -ya && \
    pip install -U spacy && \
    python -m spacy download en


RUN wget https://github.com/pytorch/fairseq/archive/v0.5.0.tar.gz && tar -zxvf v0.5.0.tar.gz
RUN cd /workspace/fairseq-0.5.0 && pip install -r requirements.txt
ENV LANG C.UTF-8
RUN cd /workspace/fairseq-0.5.0 && python setup.py build && python setup.py develop

RUN pip install pytorch-pretrained-bert
RUN pip install pytorch-transformers==1.1.0

RUN apt-get update
RUN sudo apt-get install vim screen -y
RUN apt-get install expat
RUN apt-get install libexpat-dev -y

RUN cpan install XML::Parser
RUN cpan install XML::Parser::PerlSAX
RUN cpan install XML::DOM

RUN pip install pyrouge==0.1.3

RUN if [ -d apex ]; then rm -Rf apex; fi
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && git reset --hard 453eefa56454142f8fc788478ad511973cc0fe1b && python setup.py install --cuda_ext --cpp_ext


