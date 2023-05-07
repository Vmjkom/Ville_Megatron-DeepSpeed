#!/bin/bash

DIR=/scratch/project_462000185/ville/megatron_repos/Ville_Megatron-DeepSpeed/logs/*.out

for f in $DIR
do
    egrep 'validation loss at iteration' $f | \
    perl -pe 's/.*?validation loss at iteration (\d+) \| lm loss value: (\S+).*/$1\t$2/' \
    >> $(basename -s .out $f)_loss.txt
done