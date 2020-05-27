while true
do
 export CUDA_VISIBLE_DEVICES=3
 python3 label_server.py
 echo "error" >> result.log
done
