mkdir /mnt/pred/tmp
python /workspace/pred_simple.py -i $1 -o $2 -mode 'pixel' -region 'abdom' -workspace '/workspace'
rm -r /mnt/pred/tmp