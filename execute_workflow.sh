if [ "$#" -ne 2 ]; then
  echo 'Usage: bash execute_workflow.sh <model>(regression/neural_network/neural_network_tensorflow) <method>(hold_out/cross_validation)'
  exit 1
fi

echo 'Sorting vip_.csv...'
# (head -n 1 vip_.csv && tail -n $(expr $(wc -l vip_.csv | cut -d ' ' -f 1) - 1) vip_.csv | sort) > vip_sorted_.csv
echo 'Saved sorted vip_.csv to vip_sorted_.csv'

echo 'Cleaning vip_sorted_.csv...'
# python3 clean_vip_sorted_.csv
echo 'Saved cleaned vip_sorted_.csv to vip_cleaned_.csv'

echo 'Cleaning d1.csv...'
# python3 clean_d1.py
echo 'Saved cleaned d1.csv to d1_cleaned.csv'

echo 'Cleaning idp.csv...'
# python3 clean_idp.py
echo 'Saved cleaned idp_cleaned.csv'

model=$1
method=$2
echo 'Model -' $model
echo 'Method -' $method

if [ $model == 'regression' ]; then
    if [ $method == 'hold_out' ]; then
        echo 'Training using the regression model with the hold out method...'
        # python3 train_sbp_regression.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the regression model...'
        # python3 predict_sbp_regression_2.py
        echo 'Predicted/Tested successfully'
    elif [ $method == 'cross_validation' ]; then
        echo 'Training using the regression model with the cross validation method...'
        # python3 train_sbp_regression_2.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the regression model...'
        # python3 predict_sbp_regression_3.py
        echo 'Predicted/Tested successfully'
    else
        echo 'Command line option for method incorrect, use hold_out/cross_validation'
        exit 1
    fi
elif [ $model == 'neural_network' ]; then
    if [ $method == 'hold_out' ]; then
        echo 'Training using the neural network model with the hold out method...'
        # python3 train_sbp_neural_network.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the neural network model...'
        # python3 predict_sbp_neural_network.py
        echo 'Predicted/Tested successfully'
    elif [ $method == 'cross_validation' ]; then
        echo 'Training using the neural network model with the cross validation method...'
        # python3 train_sbp_neural_network_2.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the neural network model...'
        # python3 predict_sbp_neural_network_2.py
        echo 'Predicted/Tested successfully'
    else
        echo 'Command line option for method incorrect, use hold_out/cross_validation'
        exit 1
    fi
elif [ $model == 'neural_network_tensorflow' ]; then
    echo 'Execute this model using both hold out and cross validation methods on Google Colab to make use of a GPU'

    if [ $method == 'hold_out' ]; then
        echo 'Training using the neural network tensorflow model with the hold out method on a CPU...'
        # python3 train_sbp_nn_tensorflow.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the neural network tensorflow model...'
        # python3 predict_sbp_nn_tensorflow.py
        echo 'Predicted/Tested successfully'
    elif [ $method == 'cross_validation' ]; then
        echo 'Training using the neural network tensorflow model with the cross validation method on a CPU...'
        # python3 train_sbp_nn_tensorflow_2.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the neural network tensorflow model...'
        # python3 predict_sbp_nn_tensorflow_2.py
        echo 'Predicted/Tested successfully'
    else
        echo 'Command line option for method incorrect, use hold_out/cross_validation'
        exit 1
    fi
else
    echo 'Command line option for model incorrect, use regression/neural_network/neural_network_tensorflow'
fi