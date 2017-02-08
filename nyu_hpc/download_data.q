#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=20:00:00
#PBS -l mem=2GB
#PBS -j oe
#PBS -N DSB_downloadData

cd $SCRATCH

cd lung-cancer-detector/data

cd sample

curl -sS 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/6004/sample_images.7z?sv=2015-12-11&sr=b&sig=LVG7MTZf4pm6hcZmG6Sdy%2Fz57jnExaDmJUUn56Ax1fc%3D&se=2017-01-28T19%3A06%3A57Z&sp=r' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8,en-GB;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/data-science-bowl-2017/data' -H 'Connection: keep-alive' --compressed > sample.7z

cd ../stage1

curl -sS 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/6004/stage1.7z?sv=2015-12-11&sr=b&sig=rpkP7jleG9Y6rx5xok51AtO%2BMpfmePwgjQrIrX4XHxs%3D&se=2017-01-28T19%3A15%3A56Z&sp=r' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8,en-GB;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/data-science-bowl-2017/data' -H 'Connection: keep-alive' --compressed > stage1.7z

curl -sS 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/6004/data_password.txt.zip?sv=2015-12-11&sr=b&sig=tI%2FPTmBrZK8xr9HREE9Z6V9RSk7dUYYJrp38HRPazi8%3D&se=2017-01-28T19%3A16%3A16Z&sp=r' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8,en-GB;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/data-science-bowl-2017/data' -H 'Connection: keep-alive' --compressed > data_password.txt.zip

curl -sS 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/6004/stage1_sample_submission.csv.zip?sv=2015-12-11&sr=b&sig=B%2BMRHfb7KZBtl9O%2FOI21ZFLSz7SDDJb5U%2BraJo%2FMRBg%3D&se=2017-01-28T19%3A16%3A44Z&sp=r' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8,en-GB;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/data-science-bowl-2017/data' -H 'Connection: keep-alive' --compressed > stage1_sample_submission.csv.zip

curl -sS 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/6004/stage1_labels.csv.zip?sv=2015-12-11&sr=b&sig=ItjYmhNhi42Yqn5ouGiHHkHvt46vo%2FdlDxhf3IzpMHk%3D&se=2017-01-28T19%3A17%3A16Z&sp=r' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8,en-GB;q=0.6' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/data-science-bowl-2017/data' -H 'Connection: keep-alive' --compressed > stage1_labels.csv
