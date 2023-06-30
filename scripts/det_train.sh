CONFIG='./cfgs/detections/fasterrcnn.yaml'

cd detections
python train.py --cfg_file=$CONFIG
