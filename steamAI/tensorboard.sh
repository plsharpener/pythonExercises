#! /bin/bash
function rand() {
    min=$1
	max=$(($2-$min+1))
	num=$(($RANDOM+1000000000000))
    echo $(($num%$max+$min))
}

rnd=$(rand 3000 12000)
#echo $rnd
tensorboard --logdir logs --host 0.0.0.0 --port $rnd --reload_interval 3
#tensorboard --logdir ../logdir --host 0.0.0.0 --port 1234 --reload_interval 3
#tensorboard --logdir ../logdir/$1 --host 0.0.0.0 --port $rnd --reload_interval 3
