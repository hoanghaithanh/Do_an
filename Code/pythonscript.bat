D:
cd D:\Shool_WorkSpace\Do_An_Chuan\Code\
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 9
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 8
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 7
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 6
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 5
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 4

rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 9
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 8
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 7
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 6
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 5
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 4

rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 12 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 12 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9

rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 10 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 10 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9

rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9

rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 8 --learner adam --top_Number 9
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 8 --learner adam --top_Number 9

rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 7 --learner adam --top_Number 9
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 7 --learner adam --top_Number 9

rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 6 --learner adam --top_Number 9
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 6 --learner adam --top_Number 9

rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 5 --learner adam --top_Number 9
rem python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 5 --learner adam --top_Number 9

rem #change mf_factor

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 10 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 10 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2

rem #change layers

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [96,48,24,12] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [96,48,24,12] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [120,60,30,15] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [120,60,30,15] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [144,72,36,18] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [144,72,36,18] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5

rem #change num_train_neg

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 9 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 15 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 9 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 15 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 9 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 20 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 9 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 20 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 9 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 25 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 9 --layers [72,36,18,9] --reg_mf 0 --num_train_neg 25 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8

rem #change num_test_neg

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [96,48,24,12] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 9
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [96,48,24,12] --reg_mf 0 --num_train_neg 10 --num_test_neg 1000 --lr 0.001 --learner adam --seed 9

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [96,48,24,12] --reg_mf 0 --num_train_neg 10 --num_test_neg 1500 --lr 0.001 --learner adam --seed 10
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [96,48,24,12] --reg_mf 0 --num_train_neg 10 --num_test_neg 1500 --lr 0.001 --learner adam --seed 10

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [96,48,24,12] --reg_mf 0 --num_train_neg 10 --num_test_neg 2000 --lr 0.001 --learner adam --seed 11
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 12 --layers [96,48,24,12] --reg_mf 0 --num_train_neg 10 --num_test_neg 2000 --lr 0.001 --learner adam --seed 11

pause