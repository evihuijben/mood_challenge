mkdir /mnt/pred/tmp
python /workspace/pred_simple.py -i $1 -o $2 -mode 'sample' -region 'brain' -workspace '/workspace'


echo "Finished sh file, files in /mnt/pred:"
ls /mnt/pred

echo "Removing tmp dir /mnt/pred/tmp"
rm -r /mnt/pred/tmp
echo "Files after deling tmp dir"
ls /mnt/pred
