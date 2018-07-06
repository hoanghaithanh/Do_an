D:
cd D:\Shool_WorkSpace\Do_An_Chuan\Code\

rem #change mf_factor

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 16 --layers [126,64,32,16] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 32 --layers [252,128,64,32] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 64 --layers [510,256,128,64] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3

python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 16 --layers [128,64,32,16] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 32 --layers [256,128,64,32] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 64 --layers [512,256,128,64] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3

python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 16  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 32  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 64  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3

python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 16 --layers [128,64,32,16] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 32 --layers [256,128,64,32] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 64 --layers [512,256,128,64] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3

rem #change top_number

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0 --top_number 1
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1 --top_number 2
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2 --top_number 3
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3 --top_number 4
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4 --top_number 5
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5 --top_number 6
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6 --top_number 7
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7 --top_number 8
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8 --top_number 9


python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0 --top_number 1
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1 --top_number 2
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2 --top_number 3
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3 --top_number 4
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4 --top_number 5
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5 --top_number 6
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6 --top_number 7
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7 --top_number 8
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8 --top_number 9


python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0 --top_number 1
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1 --top_number 2
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2 --top_number 3
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3 --top_number 4
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4 --top_number 5
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5 --top_number 6
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6 --top_number 7
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7 --top_number 8
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8 --top_number 9

python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0 --top_number 1
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1 --top_number 2
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2 --top_number 3
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3 --top_number 4
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4 --top_number 5
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5 --top_number 6
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6 --top_number 7
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7 --top_number 8
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8 --top_number 9

rem #change negative_training_sample

python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 1 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 2 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 3 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 5 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 6 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 7 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 8 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7
python ./Code_pure_TF_Added_FT/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [63,32,16,8] --reg_mf 0 --num_train_neg 9 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8


python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 1 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0 
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 2 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1 
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 3 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2 
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3 
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 5 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4 
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 6 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5 
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 7 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6 
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 8 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7
python ./Code_TF_Original/neuMF_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --num_train_neg 9 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8


python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 1 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0 
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 2 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1 
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 3 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2 
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3 
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 5 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 6 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5 
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 7 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6 
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 8 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7 
python ./Code_TF_Original/gmf_pure_tf_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8  --reg_mf 0 --num_train_neg 9 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8 

python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 1 --num_test_neg 1000 --lr 0.001 --learner adam --seed 0
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 2 --num_test_neg 1000 --lr 0.001 --learner adam --seed 1
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 3 --num_test_neg 1000 --lr 0.001 --learner adam --seed 2
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 4 --num_test_neg 1000 --lr 0.001 --learner adam --seed 3
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 5 --num_test_neg 1000 --lr 0.001 --learner adam --seed 4 
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 6 --num_test_neg 1000 --lr 0.001 --learner adam --seed 5 
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 7 --num_test_neg 1000 --lr 0.001 --learner adam --seed 6
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 8 --num_test_neg 1000 --lr 0.001 --learner adam --seed 7
python ./Code_TF_Original/mlp_high_API.py --path ../Data/exp_ml-20m_1/ --dataset exp_ml-20m_1 --epochs 30 --batch_size 128 --num_factors 8 --layers [64,32,16,8] --num_train_neg 9 --num_test_neg 1000 --lr 0.001 --learner adam --seed 8

pause