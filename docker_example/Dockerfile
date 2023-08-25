#FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# copy files
ADD scripts /workspace/
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
#RUN pip install nibabel

COPY requirements.txt .
RUN pip install -r requirements.txt
