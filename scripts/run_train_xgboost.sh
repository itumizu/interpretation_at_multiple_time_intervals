if [ $# -ne 1 ]; then
  exit 1
fi

dir_path=`dirname $1`
echo $dir_path

date=$(date '+%Y-%m-%d_%H-%M-%S')

comment=""

current_dir=`pwd`

while read line
do  
    if [[ $line != "#"* ]]
    then
        DIR=$line
        echo $comment
        dirs=`find $DIR -type f -name "*.yml"`
        log_date=`basename $(dirname $DIR)`"/"`basename $DIR`
        log_dir="$current_dir/logs/xgboost/"$log_date"/"

        mkdir -p $log_dir

        for dir in $dirs
        do
            filename=`basename $dir`
            dirname=`dirname $dir`
            test_fold_number=`basename $dirname`

            log_path=$log_dir"xgboost_"$filename"_"$test_fold_number".log"

            echo "Fold Number: "$test_fold_number
            echo "yml :"$dir
            echo "log :" $log_path
            
            nohup python main_train_xgboost.py --config-name $dir > $log_path &
            sleep 5s;

        done

        comment=""

    else
        comment=$line
    fi

done < $1