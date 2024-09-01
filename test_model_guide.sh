# A Quick Testing Script Guide for the Modal Contrastive Learning based Text Image Translation Model.
# Please update the corresponding path and hyper-parameters before running the code in your own environment.
echo 'Please update the corresponding path and hyper-parameters before running the code in your own environment!'

code_path=${code_path}/METIMT_Codes/evaluate.py
src_max_len=${max_length_of_source_language}
tgt_max_len=${max_length_of_source_language}
let img_width=${src_max_len}*4      # To make the sequence length of image features and text features consistent
batch_size=${batch_size}
task_name=modal_contrastive_learning_timt
model_path=${path_of_saved_path}
exp_name=${name_of_model_setting}   # Finally, the model is saved in ${model_path}/${exp_name}/
model_name=best_bleu    # or best_valid, or best_accuracy, or iter.
saved_iteration=final   # final or specific saved step.

test_image_path=${path_of_testing_images}
decoded_path=${path_of_decoded_results}

# Path of Vocabulary | txt file.
vocab_src=${path_of_source_language_vocabulary}
vocab_tgt=${path_of_target_language_vocabulary}

echo 'Remove Previous Decoded Results.'
rm ${decoded_path}

echo 'Start to Decode ...'
${python_path} ${code_path} \
--task ${task_name} \
--imgW ${img_width} --rgb \
--image_folder ${test_image_path} \
--src_vocab ${vocab_src} --tgt_vocab ${vocab_tgt} \
--src_batch_max_length ${src_max_len} \
--tgt_batch_max_length ${tgt_max_len} \
--batch_size ${batch_size} \
--saved_model ${model_path}/${exp_name}/${model_name} \
--saved_iter ${saved_iteration} \
--data_format pic \
--tgt_output ${decoded_path} 

echo 'Finished Testing.'
