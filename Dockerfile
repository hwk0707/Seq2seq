FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-py3
ADD . /
WORKDIR /

CMD ["sh", "run.sh"]