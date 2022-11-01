
CUDA_VISIBLE_DEVICES=0
do_train=true
do_test=false

for nli in cnli
do
    for model in bert roberta ernie
    do
        mkdir ./checkpoints_${model}
        save_dir="./checkpoints_${model}/${nli}"
        mkdir $save_dir
        
        if $do_train
        then
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
                --do_train \
                --model_name_or_path ${model} \
                --save_dir $save_dir \
                --train_file ./data/${nli}/train.tsv \
                --dev_file ./data/${nli}/dev.tsv \
                --num_classes 2 \
                --epochs 2 \
                --batch_size 128 \
                --learning_rate 2e-5 \
                --logging_steps 500 \
                | tee $save_dir/log_train.txt
        fi
        
        if $do_test
        then
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
                --do_test \
                --model_name_or_path ${model} \
                --init_from_ckpt $save_dir/best_model/model_state.pdparams \
                --test_file ./data/cdconv/2class_test.tsv \
                --num_classes 2 \
                --sentence_pair \
                | tee $save_dir/log_test.txt
        fi

    done
done
