import sys
import importlib
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--SRN', required=True)

args = parser.parse_args()
subname = args.SRN

try: mymodule = importlib.import_module(subname)
except:
    print("Rename your written program as YOUR_SRN.py and run python3.7 SampleTest.py --SRN YOUR_SRN ")
    sys.exit()


get_selected_attribute = mymodule.get_selected_attribute
get_information_gain = mymodule.get_information_gain
get_avg_info_of_attribute = mymodule.get_avg_info_of_attribute
get_entropy_of_dataset = mymodule.get_entropy_of_dataset


def test_case():
    outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
    temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
    humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
    windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
    play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')
    dataset = {'outlook': outlook, 'temp': temp, 'humidity': humidity, 'windy': windy, 'play': play}
    df = pd.DataFrame(dataset, columns=['outlook', 'temp', 'humidity', 'windy', 'play'])
    
    # get_entropy_of_dataset
    get_entropy_of_dataset_testcases = [
        (df)
    ]
    get_entropy_of_dataset_solutions = [
        (0.938, 0.942)
    ]
    
    # get_avg_info_of_attribute
    get_avg_info_of_attribute_testcases = [
        (df, 'outlook'),
        (df, 'temp')
    ]
    get_avg_info_of_attribute_solutions = [
        (0.691, 0.695),
        (0.908, 0.914)
    ]
    
    # get_selected_attribute_testcases = [
    #     (df)
    # ]
    # get_selected_attribute_solutions = [
        
    # ]


    print("Testing the function get_entropy_of_dataset...")
    for i in range(len(get_entropy_of_dataset_testcases)):
        try:
            res = get_entropy_of_dataset(get_entropy_of_dataset_testcases[i])
            if res >= get_entropy_of_dataset_solutions[i][0] and res <= get_entropy_of_dataset_solutions[i][1]:
                print("Test Case", "{0:2d}".format(i), "for  get_entropy_of_dataset   \033[92mPASSED\033[0m")
            else:
                print("Test Case 1 for  get_entropy_of_dataset   \033[91mFAILED\033[0m")
        except Exception as e:
            print("Test Case", "{0:2d}".format(i), "for  get_entropy_of_dataset   \033[91mFAILED\033[0m due to ", e)
    print("-------------------------------------------------\n")

    print("Testing the function get_avg_info_of_attribute...")
    for i in range(len(get_avg_info_of_attribute_testcases)):
        try:
            res = get_avg_info_of_attribute(*get_avg_info_of_attribute_testcases[i])
            if res >= get_avg_info_of_attribute_solutions[i][0] and res <= get_avg_info_of_attribute_solutions[i][1]:
                print("Test Case", "{0:2d}".format(i), "for get_avg_info_of_attribute \033[92mPASSED\033[0m")
            else:
                print("Test Case", "{0:2d}".format(i), "for get_avg_info_of_attribute \033[91mFAILED\033[0m")
        except Exception as e:
            print("Test Case", "{0:2d}".format(i), "for get_avg_info_of_attribute \033[91mFAILED\033[0m due to ", e)
    print("-------------------------------------------------\n")

    print("Testing the function get_selected_attribute...")
    #for i in range(len(get_selected_attribute_testcases)):
    try:
        res = get_selected_attribute(df)
        dictionary = res[0]
        flag = (dictionary['outlook'] >= 0.244 and dictionary['outlook'] <= 0.248) and (dictionary['temp'] >= 0.0292 and dictionary['temp'] <= 0.0296) and (dictionary['humidity'] >= 0.150 and dictionary['humidity'] <= 0.154) and (dictionary['windy'] >= 0.046 and dictionary['windy'] <= 0.05) and (res[1] == 'outlook')
        if flag:
            print("Test Case", "{0:2d}".format(i), "for get_selected_attribute    \033[92mPASSED\033[0m")
        else:
            print("Test Case", "{0:2d}".format(i), "for get_selected_attribute    \033[91mFAILED\033[0m")
    except Exception as e:
        print("Test Case", "{0:2d}".format(i), "for get_selected_attribute    \033[91mFAILED\033[0m due to ", e)
    print("-------------------------------------------------\n")


if __name__ == "__main__":
    test_case()