
CUDA_VISIBLE_DEVICES=0
do_train=true
do_test=true

for nli in cnli
do
    for num in 4
    do
        for model in bert
        do
            mkdir ./checkpoints_${model}
            mkdir ./checkpoints_${model}/cdconv_hierarchical
            mkdir ./checkpoints_${model}/cdconv_hierarchical/${num}class_${nli}_intra
            mkdir ./checkpoints_${model}/cdconv_hierarchical/${num}class_${nli}_role
            mkdir ./checkpoints_${model}/cdconv_hierarchical/${num}class_${nli}_hist
            
            for seed in 23 42 133 233
            do

                for type in intra role hist
                do
                    save_dir="./checkpoints_${model}/cdconv_hierarchical/${num}class_${nli}_${type}/${seed}"
                    mkdir $save_dir
                    
                    if $do_train
                    then
                        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
                            --do_train \
                            --model_name_or_path ${model} \
                            --init_from_ckpt ./checkpoints_${model}/${nli}/best_model/model_state.pdparams \
                            --save_dir $save_dir \
                            --train_file ./data/cdconv_hierarchical/${num}class_train_${type}.tsv \
                            --dev_file ./data/cdconv_hierarchical/${num}class_dev_${type}.tsv \
                            --num_classes 2 \
                            | tee $save_dir/log_train.txt
                    fi
                    
                    if $do_test
                    then
                        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
                            --do_test \
                            --model_name_or_path ${model} \
                            --init_from_ckpt $save_dir/best_model/model_state.pdparams \
                            --test_file ./data/cdconv_hierarchical/${num}class_test_${type}.tsv \
                            --num_classes 2 \
                            | tee $save_dir/log_test.txt
                    fi
                
                done
            done
        done 
    done
done