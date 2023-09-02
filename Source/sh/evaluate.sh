#!/bin/bash
lang=${1}
generator=${2}
discriminator="CodeBERT"
if [ ${discriminator} = "CodeBERT" ]; then
    script_name = "main.py"
else
    script_name = "main_GCB.py"
fi

python ${script_name}
 --do_train_generator \
 --do_train_discriminator \
 --do_test \
 --language ${lang} \
 --train_filename ./Dataset/codeSearchNet/${lang}/train.jsonl \
 --dev_filename ./Dataset/codeSearchNet/${lang}/test.jsonl \
 --test_filename ./Dataset/codeSearchNet/${lang}/test.jsonl \
 --output_dir models/finetuning/${lang}/v0 \
 --gen_lr 2e-5 \
 --dis_lr 1e-5 \
 --train_batch_size 64 \
 --gradient_accumulation_steps 2 \
 --eval_batch_size 32 \
 --gpu_id 0 \
 --load_generator_path models/pretraining/${generator}_${lang}/pytorch_model.bin \
 --load_discriminator_path models/pretraining/${discriminator}_${lang}/pytorch_model.bin \
 --dk_epochs 1 \
 --max_source_length 128 \
 --max_dis_seq_length 200 \
 --max_target_length 256 \
 --beam_size 10 \
 --n_rollout 1 \
 --similarity_filename ranks/${lang}_similarity_ranks.bin