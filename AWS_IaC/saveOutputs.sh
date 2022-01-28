#!/bin/bash


function_name () {
    echo -n \" >> $2;
    echo -n ${1/' = '/'" : '} >> $2;
    echo , >> $2;
}

echo { > aws_resources.txt;

terraform output| while read -r line; do function_name "$line" "aws_resources.txt"; done;

echo } >> aws_resources.txt;