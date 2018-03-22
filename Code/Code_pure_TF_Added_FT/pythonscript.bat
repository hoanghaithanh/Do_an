D:
cd D:\Shool_WorkSpace\Do_An_Chuan\Code\Code_pure_TF_Added_FT
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 9
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 8
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 7
rem python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 6
python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 5
python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 4

python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 9
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 8
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 7
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 6
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 5
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [72,36,18,9] --num_neg 8 --learner adam --top_Number 4

python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 12 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 12 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9

python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 10 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 10 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9

python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [96,48,24,12] --num_neg 8 --learner adam --top_Number 9

python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 8 --learner adam --top_Number 9
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 8 --learner adam --top_Number 9

python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 7 --learner adam --top_Number 9
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 7 --learner adam --top_Number 9

python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 6 --learner adam --top_Number 9
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 6 --learner adam --top_Number 9

python neuMF_pure_tf.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 5 --learner adam --top_Number 9
python neuMF_pure_tf_no_ft.py --dataset ml-20m --epochs 20 --batch_size 256 --num_factors 8 --layers [48,24,12,6] --num_neg 5 --learner adam --top_Number 9

pause