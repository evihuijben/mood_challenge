mkdir /mnt/pred/tmp
python /workspace/pred_simple.py -i $1 -o $2 -mode 'sample' -region 'brain' -workspace '/workspace'
