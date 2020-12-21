#!/bin/bash

if [ ! -d ./logs ]
then
    echo "Create directory ./logs"
    mkdir ./logs
fi


if [ ! -d /mnt/open_lth_data ]
then
    echo "Create directory /mnt/open_lth_data"
    mkdir /mnt/open_lth_data
fi

chown ubuntu:ubuntu -R /mnt
