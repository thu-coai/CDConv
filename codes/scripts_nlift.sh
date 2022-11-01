
CUDA_VISIBLE_DEVICES=0
do_train=true
do_test=true

for nli in cnli
do
    for num in 2 4
    do
        for model in bert roberta ernie
        do
            mkdir ./checkpoints_${model}
            mkdir ./checkpoints_${model}/cdconv
            mkdir ./checkpoints_${model}/cdconv/${num}class_${nli}_flatten
            mkdir ./checkpoints_${model}/cdconv/${num}class_${nli}_sentpair
            
            for seed in 23 42 133 233
            do
                # w/ ctx
                save_dir="./checkpoints_${model}/cdconv/${num}class_${nli}_flatten/${seed}"
                mkdir $save_dir
                
                if $do_train
                then
                    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
                        --do_train \
                        --model_name_or_path ${model} \
                        --init_from_ckpt ./checkpoints_${model}/${nli}/best_model/model_state.pdparams \
                        --save_dir $save_dir \
                        --train_file ./data/cdconv/${num}class_train.tsv \
                        --dev_file ./data/cdconv/${num}class_dev.tsv \
                        --num_classes ${num} \
                        | tee $save_dir/log_train.txt
                fi
                
                if $do_test
                then
                    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
                        --do_test \
                        --model_name_or_path ${model} \
                        --init_from_ckpt $save_dir/best_model/model_state.pdparams \
                        --test_file ./data/cdconv/${num}class_test.tsv \
                        --num_classes ${num} \
                        | tee $save_dir/log_test.txt
                fi
                
                # w/o ctx
                save_dir="./checkpoints_${model}/cdconv/${num}class_${nli}_sentpair/${seed}"
                mkdir $save_dir
                
                if $do_train
                then
                    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
                        --do_train \
                        --model_name_or_path ${model} \
                        --init_from_ckpt ./checkpoints_${model}/${nli}/best_model/model_state.pdparams \
                        --save_dir $save_dir \
                        --train_file ./data/cdconv/${num}class_train.tsv \
                        --dev_file ./data/cdconv/${num}class_dev.tsv \
                        --num_classes ${num} \
                        --sentence_pair \
                        | tee $save_dir/log_train.txt
                fi
                
                if $do_test
                then
                    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
                        --do_test \
                        --model_name_or_path ${model} \
                        --init_from_ckpt $save_dir/best_model/model_state.pdparams \
                        --test_file ./data/cdconv/${num}class_test.tsv \
                        --num_classes ${num} \
                        --sentence_pair \
                        | tee $save_dir/log_test.txt
                fi

            done
        done 
    done
done