# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os, subprocess
import argparse

def __validate_folders(args):
    if not args.train:
        if os.path.isdir(output_dir):
            raise Exception(output_dir + " is an existing folder. Please either delete it or rename it.")
    
    if not os.path.isdir(lasertagger_dir):
        raise Exception("LaserTagger not found.")
    if not os.path.isdir(os.path.expanduser(lasertagger_dir + "/bert")):
        raise Exception("Bert not found inside the LaserTagger folder.")
    
    if not os.path.isdir(os.path.expanduser(bert_dir)):
        raise Exception("Pretrained Bert model not found.")

def __validate_files(file):
    if not os.path.isfile(os.path.expanduser(file)):
        raise Exception("File " + str(file) + " not found/")

def __set_parameters(args):
    global vocab_size
    vocab_size = str(args.vocab_size)
    global train_batch_size
    train_batch_size = str(args.train_batch_size)
    global learning_rate
    learning_rate = str(args.learning_rate)
    global num_train_epochs
    num_train_epochs = str(args.num_train_epochs)
    global warmup_proportion
    warmup_proportion = str(args.warmup_proportion)
    
    global output_dir
    output_dir = os.path.expanduser(args.model_output_dir)
    global bert_dir
    bert_dir = os.path.expanduser(args.abs_path_to_bert)
    global lasertagger_dir
    lasertagger_dir = os.path.expanduser(args.abs_path_to_lasertagger)
    
    global training_file
    training_file =  os.path.expanduser(args.training_file)
    global tuning_file
    tuning_file =  os.path.expanduser(args.tuning_file)
    
def __preprocess(args):
    print("------ Running vocab optimization ------")
    output_file = output_dir + "/label_map.txt"
    
    with open(os.path.expanduser(args.training_file), 'r') as f:
        num_training = sum(1 for _ in f)
    
    if args.max_input_examples:
        if args.max_input_examples <= num_training:
            max_input_examples = str(args.max_input_examples)
        else:
            raise Exception("max input example > training data count")
    else:
        max_input_examples = str(int(num_training/3))
    
    print("number of maximum input is", max_input_examples)
    print("number of vocab size is", vocab_size)
        
    subprocess.call(("python phrase_vocabulary_optimization.py --input_file=" + training_file + 
                     " --input_format=wikisplit" +
                     " --vocabulary_size=" + vocab_size + 
                     " --max_input_examples=" + max_input_examples + 
                     " --output_file=" + output_file).split(), 
                    cwd=lasertagger_dir)
    
    print("------ Running preprocessing ------")
    subprocess.call(("python preprocess_main.py --input_file=" + tuning_file + 
                     " --input_format=wikisplit" +
                     " --output_tfrecord=" + output_dir + "/tune.tf_record" + 
                     " --label_map_file=" + output_file + 
                     " --vocab_file=" + bert_dir + "/vocab.txt"
                     " --output_arbitrary_targets_for_infeasible_examples=true").split() , 
                    cwd=lasertagger_dir)
    
    subprocess.call(("python preprocess_main.py --input_file=" + training_file + 
                     " --input_format=wikisplit" +
                     " --output_tfrecord=" + output_dir + "/train.tf_record" + 
                     " --label_map_file=" + output_file + 
                     " --vocab_file=" + bert_dir + "/vocab.txt"
                     " --output_arbitrary_targets_for_infeasible_examples=false").split() , 
                    cwd=lasertagger_dir)
    
    
def __training(args):
    print("------ Start training ------")
    f = open(output_dir + "/train.tf_record.num_examples.txt","r")
    num_train_examples = f.read()
    f = open(output_dir + "/tune.tf_record.num_examples.txt","r")
    num_tune_examples = f.read()
    
    print("training batch size is", train_batch_size)
    print("learning rate is", learning_rate)
    print("number of training epochs is", num_train_epochs)
    print("warm up proportion is", warmup_proportion)
    
    subprocess.call(("python run_lasertagger.py --training_file=" + output_dir + "/train.tf_record" + 
                     " --eval_file=" + output_dir + "/tune.tf_record" + 
                     " --label_map_file=" + output_dir + "/label_map.txt" + 
                     " --model_config_file=" + lasertagger_dir + "/configs/lasertagger_config.json" + 
                     " --output_dir=" + output_dir + "/model" + 
                     " --init_checkpoint=" + bert_dir + "/bert_model.ckpt" + 
                     " --do_train=true" + 
                     " --do_eval=true" + 
                     " --save_checkpoints_steps=500" + 
                     " --num_train_examples=" + num_train_examples +
                     " --num_eval_examples=" + num_tune_examples + 
                     " --train_batch_size=" + train_batch_size +
                     " --learning_rate=" + learning_rate + 
                     " --num_train_epochs=" + num_train_epochs +
                     " --warmup_proportion" + warmup_proportion).split() , 
                    cwd=lasertagger_dir)

    print("------ Completed training ------")
    print("------ Start exporting ------")
    subprocess.call(("python run_lasertagger.py" + 
                     " --label_map_file=" + output_dir + "/label_map.txt" + 
                     " --model_config_file=" + lasertagger_dir + "/configs/lasertagger_config.json" + 
                     " --output_dir=" + output_dir + "/model" + 
                     " --do_export=true" + 
                     " --export_path=" + output_dir + "/export").split() , 
                    cwd=lasertagger_dir)
    os.rename(output_dir + "/export/" + os.listdir(output_dir + "/export")[0], output_dir + "/export/" + "export_model")

def __export_to_bucket(args):
    print("------ Exporting to bucket ------")
    folder_name = output_dir.split("/")[-1]
    subprocess.call(("gsutil -m cp -r " + output_dir + "/export/export_model gs://trained_models_yechen/" + folder_name + "/").split(), cwd=os.path.expanduser("~"))
    subprocess.call(("gsutil -m cp -r " + output_dir + "/label_map.txt gs://trained_models_yechen/" + folder_name + "/").split(), cwd=os.path.expanduser("~"))
    
def main(model_dir, lasertagger_path, bert_path, vocab_size):
    os.environ["VOCAB_SIZE"] = str(vocab_size)
    os.environ["OUTPUT_FILE"] = "./dataset_output_"+ str(vocab_size) + "/label_map.txt"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_output_dir", help="the directory of the model output")
    parser.add_argument("abs_path_to_lasertagger", help="absolute path to the folder where the lasertagger scripts are located")
    parser.add_argument("abs_path_to_bert", help="absolute path to the folder where the pretrained BERT is located")
    parser.add_argument("training_file", help="path to training samples")
    parser.add_argument("tuning_file", help="path to tuning samples")
    
    parser.add_argument("-vocab_size", type=int, help="vocab size", default=500)
    parser.add_argument("-train_batch_size", type=int, help="batch size during training", default=32)
    parser.add_argument("-learning_rate", type=float, help="The initial learning rate for Adam", default=3e-5)
    parser.add_argument("-num_train_epochs", type=int, help="Total number of training epochs to perform", default=3)
    parser.add_argument("-warmup_proportion", type=float, help="Proportion of training to perform linear learning rate warmup for", default=0.1)
    parser.add_argument("-max_input_examples", type=int, help="number of training examples to use in the vocab optimization")
    parser.add_argument("-train", action="store_true", help="if added, skip preprocessing and start training")
    parser.add_argument("-export", action="store_true", help="if added, skip preprocessing and training, and start exporting to bucket")
    args = parser.parse_args()
    
    __set_parameters(args)
    __validate_folders(args)
    __validate_files(args.training_file)
    __validate_files(args.tuning_file)
    
    print(args.train)
    
    if args.export:
        print("------ Skipped preprocessing and training ------")
        __export_to_bucket(args)
    elif args.train:
        print("------ Skipped preprocessing ------")
        __training(args)
        __export_to_bucket(args)
    else:
        subprocess.call(['mkdir', os.path.expanduser(args.model_output_dir)], cwd=os.path.expanduser('~'))
        print("------Made new directory", args.model_output_dir, "------")
        __preprocess(args)
        __training(args)
        __export_to_bucket(args)