ps aux | grep "main_train_xgboost.py" | grep -v grep | awk '{ print "kill -9", $2 }' | sh