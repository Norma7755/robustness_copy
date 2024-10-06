#! MNIST
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_mnist.py --AT_type Nat --model_name SSM --attack_type PGD --model-dir checkpoints/model-MNIST
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_mnist.py --AT_type Nat --model_name DSS --attack_type PGD --model-dir checkpoints/model-MNIST
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_mnist.py --AT_type Nat --model_name S5 --attack_type PGD --model-dir checkpoints/model-MNIST
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_mnist.py --AT_type Nat --model_name Mega --attack_type PGD --model-dir checkpoints/model-MNIST
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_mnist.py --AT_type Nat --model_name S6 --attack_type PGD --model-dir checkpoints/model-MNIST
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_mnist.py --AT_type Nat --model_name Transformer --attack_type PGD --model-dir checkpoints/model-MNIST


#! CIFAR10
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type Nat --model_name SSM --attack_type PGD --model-dir checkpoints/model-CIFAR10
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type Nat --model_name DSS --attack_type PGD --model-dir checkpoints/model-CIFAR10
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type Nat --model_name S5 --attack_type PGD --model-dir checkpoints/model-CIFAR10
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type Nat --model_name Mega --attack_type PGD --model-dir checkpoints/model-CIFAR10
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type Nat --model_name S6 --attack_type PGD --model-dir checkpoints/model-CIFAR10
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type Nat --model_name Transformer --attack_type PGD --model-dir checkpoints/model-CIFAR10


#! tiny imagenet
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_tinyimagenet.py --AT_type Nat --model_name SSM --attack_type PGD --model-dir checkpoints/model-tinyimagenet
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_tinyimagenet.py --AT_type Nat --model_name DSS --attack_type PGD --model-dir checkpoints/model-tinyimagenet
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_tinyimagenet.py --AT_type Nat --model_name S5 --attack_type PGD --model-dir checkpoints/model-tinyimagenet
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_tinyimagenet.py --AT_type Nat --model_name Mega --attack_type PGD --model-dir checkpoints/model-tinyimagenet
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_tinyimagenet.py --AT_type Nat --model_name S6 --attack_type PGD --model-dir checkpoints/model-tinyimagenet
srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_tinyimagenet.py --AT_type Nat --model_name Transformer --attack_type PGD --model-dir checkpoints/model-tinyimagenet