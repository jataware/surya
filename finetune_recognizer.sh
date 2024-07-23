
# BASE_DIR='/home/paperspace/zeek/'
# mkdir -p $BASE_DIR + 'data/'
# python generate_text.py --num_images 1000 --save_dir $BASE_DIR + 'data/leading_0s/'

# show improvement on leading 0s
python train_recognizer.py \
    --experiment_name 'leading_zeros_base' \
    --data_dir $BASE_DIR + 'data/leading_0s/' \
    --save_dir $BASE_DIR + 'models/leading_0s/' \
    --n_train 500

# show lack of degradation on normal data w leading 0s
python train_recognizer.py \
    --experiment_name 'leading_zeros_degrade' \
    --data_dir $BASE_DIR + 'data/leading_0s/' \
    --save_dir $BASE_DIR + 'models/leading_0s/' \
    --n_train 500

