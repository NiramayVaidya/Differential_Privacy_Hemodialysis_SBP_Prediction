if [ "$#" -ne 2 ]; then
  echo 'Usage: bash execute_workflow.sh <model>(regression/neural_network/neural_network_tensorflow) <method>(hold_out/cross_validation)'
  exit 1
fi

# Without DP
# Hold out and cross validation
# vip.csv
# vip_.csv

# With DP
# Hold out and cross validation
# vip_dp_0_1.csv
# vip_dp_1.csv
# vip_dp_2.csv
# vip_dp_0_1_.csv
# vip_dp_1_.csv
# vip_dp_2_.csv
vip_filename='vip_files/vip_dp_2_time_perturbed_.csv'

# Without DP
# Hold out and cross validation
# vip_cleaned.csv
# vip_cleaned_.csv

# With DP
# Hold out and cross validation
# vip_cleaned_dp_0_1.csv
# vip_cleaned_dp_1.csv
# vip_cleaned_dp_2.csv
# vip_cleaned_dp_0_1_.csv
# vip_cleaned_dp_1_.csv
# vip_cleaned_dp_2_.csv
vip_cleaned_filename='vip_files/vip_cleaned_dp_2_time_perturbed_.csv'

echo 'Sorting '$vip_filename'...'
# (head -n 1 $vip_filename && tail -n $(expr $(wc -l $vip_filename | cut -d ' ' -f 1) - 1) $vip_filename | sort) > vip_generic_sorted.csv
echo 'Saved sorted '$vip_filename' to vip_generic_sorted.csv'

echo 'Cleaning vip_generic_sorted.csv...'
# python3 clean_vip_generic_sorted.py
echo 'Saved cleaned vip_generic_sorted.csv to '$vip_cleaned_filename

echo 'Removing intermediate file vip_generic_sorted.csv...'
# rm vip_generic_sorted.csv
echo 'Removed vip_generic_sorted.csv'

echo 'Cleaning d1.csv...'
# python3 clean_d1.py
echo 'Saved cleaned d1.csv to d1_cleaned.csv'

echo 'Cleaning idp.csv...'
# python3 clean_idp.py
echo 'Saved cleaned idp_cleaned.csv'

model=$1
method=$2
echo 'Model - '$model
echo 'Method - '$method

if [ $model == 'regression' ]; then
    if [ $method == 'hold_out' ]; then
        echo 'Training using the regression model with the hold out method...'
        python3 train_sbp_regression.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the regression model...'
        python3 predict_sbp_regression_2.py
        echo 'Predicted/Tested successfully'
    elif [ $method == 'cross_validation' ]; then
        echo 'Training using the regression model with the cross validation method...'
        python3 train_sbp_regression_2.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the regression model...'
        python3 predict_sbp_regression_3.py
        echo 'Predicted/Tested successfully'
    else
        echo 'Command line option for method incorrect, use hold_out/cross_validation'
        exit 1
    fi
elif [ $model == 'neural_network' ]; then
    if [ $method == 'hold_out' ]; then
        echo 'Training using the neural network model with the hold out method...'
        python3 train_sbp_neural_network.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the neural network model...'
        python3 predict_sbp_neural_network.py
        echo 'Predicted/Tested successfully'
    elif [ $method == 'cross_validation' ]; then
        echo 'Training using the neural network model with the cross validation method...'
        python3 train_sbp_neural_network_2.py
        echo 'Trained successfully'

        echo 'Predicting/Testing using the neural network model...'
        python3 predict_sbp_neural_network_2.py
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